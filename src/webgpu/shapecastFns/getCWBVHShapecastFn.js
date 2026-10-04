/** @import { StructTypeNode } from 'three/webgpu' */
/** @import { BVHComputeData } from '../BVHComputeData.js' */
import { wgslTagCode, wgslTagFn } from '../nodes/WGSLTagFnNode.js';
import { bvhNodeBoundsStruct, cwbvhNodeStruct } from '../tsl/structs.js';
import { CWBVH_STACK_DEPTH } from '../tsl/constants.js';

const getCWBVHChildBounds = wgslTagFn/* wgsl */`
	fn bvh_GetCWBVHChildBounds( node: ${ cwbvhNodeStruct }, slot: u32 ) -> ${ bvhNodeBoundsStruct } {

		let exponentData = node.data[ 3 ];
		let exponent = vec3f(
			f32( i32( exponentData << 24u ) >> 24u ),
			f32( i32( exponentData << 16u ) >> 24u ),
			f32( i32( exponentData << 8u ) >> 24u )
		);
		let scale = exp2( exponent );
		let origin = vec3f(
			bitcast<f32>( node.data[ 0 ] ),
			bitcast<f32>( node.data[ 1 ] ),
			bitcast<f32>( node.data[ 2 ] )
		);
		let wordOffset = slot >> 2u;
		let shift = ( slot & 3u ) * 8u;

		let minX = ( node.data[ 6u + wordOffset ] >> shift ) & 0xffu;
		let maxX = ( node.data[ 8u + wordOffset ] >> shift ) & 0xffu;
		let minY = ( node.data[ 10u + wordOffset ] >> shift ) & 0xffu;
		let maxY = ( node.data[ 12u + wordOffset ] >> shift ) & 0xffu;
		let minZ = ( node.data[ 14u + wordOffset ] >> shift ) & 0xffu;
		let maxZ = ( node.data[ 16u + wordOffset ] >> shift ) & 0xffu;

		var result: ${ bvhNodeBoundsStruct };
		result.min[ 0 ] = origin.x + f32( minX ) * scale.x;
		result.min[ 1 ] = origin.y + f32( minY ) * scale.y;
		result.min[ 2 ] = origin.z + f32( minZ ) * scale.z;
		result.max[ 0 ] = origin.x + f32( maxX ) * scale.x;
		result.max[ 1 ] = origin.y + f32( maxY ) * scale.y;
		result.max[ 2 ] = origin.z + f32( maxZ ) * scale.z;
		return result;

	}
`;

/**
 * CWBVH traversal variant used by {@link BVHComputeData#getShapecastFn}.
 *
 * @private
 * @param {BVHComputeData} bvhData
 * @param {Object} options
 * @param {string} [options.name]
 * @param {StructTypeNode} options.shapeStruct
 * @param {StructTypeNode|null} [options.resultStruct]
 * @param {Function|null} [options.prefixFn]
 * @param {Function|null} [options.childOrderFn]
 * @param {Function} options.intersectsBoundsFn
 * @param {Function} options.intersectRangeFn
 * @param {Function|null} [options.transformShapeFn]
 * @param {Function|null} [options.transformResultFn]
 * @param {Function|null} [options.resetShapeFn]
 * @returns {Function}
 */
export function getCWBVHShapecastFn( bvhData, options ) {

	const {
		name = `bvh_cwbvh_shapecast_fn_${ Math.random().toString( 36 ).substring( 2, 7 ) }`,
		shapeStruct,
		resultStruct = null,

		prefixFn = null,
		childOrderFn = null,
		intersectsBoundsFn,
		intersectRangeFn,
		transformShapeFn = null,
		transformResultFn = null,
		resetShapeFn = null,
	} = options;

	const { nodes, leaves, transforms } = bvhData.storage;

	let prefixSnippet = '';
	if ( prefixFn ) prefixSnippet = wgslTagCode/* wgsl */`${ prefixFn }();`;

	let transformResultSnippet = '';
	if ( transformResultFn ) transformResultSnippet = wgslTagCode/* wgsl */`${ transformResultFn }( result, objectIndex );`;

	let transformShapeSnippet = '';
	if ( transformShapeFn ) transformShapeSnippet = wgslTagCode/* wgsl */`${ transformShapeFn }( &localShape, objectIndex );`;

	let resetShapeSnippet = '';
	if ( resetShapeFn ) resetShapeSnippet = wgslTagCode/* wgsl */`${ resetShapeFn }( objectIndex );`;

	let childOrderSnippet = wgslTagCode/* wgsl */`let childOrder = 0u;`;
	if ( childOrderFn ) childOrderSnippet = wgslTagCode/* wgsl */`let childOrder = ${ childOrderFn }( localShape ) & 7u;`;

	const resultPtrSnippet = resultStruct ? wgslTagCode/* wgsl */`, result: ptr<function, ${ resultStruct }>` : '';
	const resultArgSnippet = resultStruct ? ', result' : '';

	const fn = wgslTagFn/* wgsl */`
		fn ${ name }( shape: ${ shapeStruct }${ resultPtrSnippet } ) -> bool {

			${ prefixSnippet }

			var didHit = false;
			var isTLAS = true;
			var pointer: i32 = 0;
			var stack: array<u32, ${ CWBVH_STACK_DEPTH }>;
			stack[ 0 ] = 0u;

			var blasDidHit = false;
			var objectIndex = 0u;
			var localShape: ${ shapeStruct } = shape;
			var tlasReset: i32 = 0;

			loop {

				if ( ! isTLAS && tlasReset == pointer ) {

					if ( blasDidHit ) {

						blasDidHit = false;
						didHit = true;
						${ transformResultSnippet }

					}

					${ resetShapeSnippet }
					objectIndex = 0u;
					isTLAS = true;
					localShape = shape;

				}

				if ( pointer < 0 || pointer >= i32( ${ CWBVH_STACK_DEPTH } ) ) {

					break;

				}

				let entry = stack[ pointer ];
				pointer = pointer - 1;

				if ( ( entry & 0x80000000u ) != 0u ) {

					let leafIndex = entry & 0x7fffffffu;
					let leaf = ${ leaves }[ leafIndex ];

					if ( ( leaf.info & 0x80000000u ) != 0u ) {

						objectIndex = leaf.info & 0x7fffffffu;
						let transform = ${ transforms }[ objectIndex ];
						if ( transform.visible != 0u ) {

							tlasReset = pointer;
							isTLAS = false;
							blasDidHit = false;
							localShape = shape;
							${ transformShapeSnippet }

							pointer = pointer + 1;
							stack[ pointer ] = leaf.value;

						}

					} else {

						blasDidHit = ${ intersectRangeFn }( localShape, leaf.value, leaf.info${ resultArgSnippet } ) || blasDidHit;

					}

					continue;

				}

				let node = ${ nodes }[ entry ];
				${ childOrderSnippet }

				for ( var orderIndex: i32 = 7; orderIndex >= 0; orderIndex = orderIndex - 1 ) {

					let slot = u32( orderIndex ) ^ childOrder;
					let shift = ( slot & 3u ) * 8u;
					let metadata = ( node.data[ 18u + ( slot >> 2u ) ] >> shift ) & 0xffu;
					if ( metadata == 0u ) {

						continue;

					}

					let bounds = ${ getCWBVHChildBounds }( node, slot );
					if ( ${ intersectsBoundsFn }( localShape, bounds${ resultArgSnippet } ) == 0u ) {

						continue;

					}

					pointer = pointer + 1;
					if ( ( metadata & 0x80u ) != 0u ) {

						stack[ pointer ] = node.data[ 4 ] + ( metadata & 7u );

					} else {

						let leafIndex = node.data[ 5 ] + ( metadata & 7u );
						stack[ pointer ] = 0x80000000u | leafIndex;

					}

				}

			}

			return didHit;

		}
	`;

	fn.outputType = resultStruct;
	fn.functionName = name;
	return fn;

}
