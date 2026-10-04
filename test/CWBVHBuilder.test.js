import { BufferGeometry, Float32BufferAttribute, Mesh, TorusKnotGeometry, WebGPUCoordinateSystem } from 'three';
import { WGSLNodeBuilder } from 'three/webgpu';
import { context, wgslFn } from 'three/tsl';
import { MeshBVH } from '../src/core/MeshBVH.js';
import { BVHComputeData } from '../src/webgpu/BVHComputeData.js';
import { CWBVHBuilder, CWBVH_NODE_U32 } from '../src/webgpu/utils/CWBVHBuilder.js';

function getByte( array, index, slot ) {

	return ( array[ index + ( slot >> 2 ) ] >> ( ( slot & 3 ) * 8 ) ) & 0xff;

}

function getScale( packedExponent, axis ) {

	const value = ( packedExponent >> ( axis * 8 ) ) & 0xff;
	return Math.pow( 2, value > 127 ? value - 256 : value );

}

function validateAndCollectPrimitives( geometry, data, root = 0 ) {

	const nodesU32 = data.nodes;
	const nodesF32 = new Float32Array( nodesU32.buffer, nodesU32.byteOffset, nodesU32.length );
	const primitives = [];
	visit( root );
	return primitives;

	function visit( nodeIndex ) {

		const index = nodeIndex * CWBVH_NODE_U32;
		const childBase = nodesU32[ index + 4 ];
		const primitiveBase = nodesU32[ index + 5 ];

		for ( let slot = 0; slot < 8; slot ++ ) {

			const metadata = getByte( nodesU32, index + 18, slot );
			if ( metadata === 0 ) continue;

			const bounds = [];
			for ( let axis = 0; axis < 3; axis ++ ) {

				const scale = getScale( nodesU32[ index + 3 ], axis );
				const origin = nodesF32[ index + axis ];
				bounds[ axis ] = origin + getByte( nodesU32, index + 6 + axis * 4, slot ) * scale;
				bounds[ axis + 3 ] = origin + getByte( nodesU32, index + 8 + axis * 4, slot ) * scale;

			}

			if ( metadata & 0x80 ) {

				visit( childBase + ( metadata & 7 ) );

			} else {

				const leafIndex = primitiveBase + ( metadata & 7 );
				const triangleOffset = data.leaves[ leafIndex * 2 ];
				const count = data.leaves[ leafIndex * 2 + 1 ];
				for ( let i = 0; i < count; i ++ ) {

					const triangleIndex = triangleOffset + i;
					primitives.push( triangleIndex );
					validateTriangleBounds( geometry, triangleIndex, bounds );

				}

			}

		}

	}

}

function validateTriangleBounds( geometry, triangleIndex, bounds ) {

	const index = geometry.index;
	const position = geometry.attributes.position;
	for ( let axis = 0; axis < 3; axis ++ ) {

		let min = Infinity;
		let max = - Infinity;
		for ( let corner = 0; corner < 3; corner ++ ) {

			const vertex = index ? index.getX( triangleIndex * 3 + corner ) : triangleIndex * 3 + corner;
			const value = position.getComponent( vertex, axis );
			min = Math.min( min, value );
			max = Math.max( max, value );

		}

		expect( bounds[ axis ] ).toBeLessThanOrEqual( min );
		expect( bounds[ axis + 3 ] ).toBeGreaterThanOrEqual( max );

	}

}

describe( 'CWBVHBuilder', () => {

	it( 'packs conservative bounds and every primitive exactly once', () => {

		const geometry = new TorusKnotGeometry( 1, 0.3, 64, 8 );
		const bvh = new MeshBVH( geometry, { targetLeafSize: 1 } );
		const builder = new CWBVHBuilder();
		const info = builder.add( bvh._roots[ 0 ], 0, ( offset, count ) => ( { value: offset, meta: count } ) );
		const data = builder.build();
		const primitives = validateAndCollectPrimitives( geometry, data, info.root );

		expect( primitives.sort( ( a, b ) => a - b ) ).toEqual( Array.from( { length: geometry.index.count / 3 }, ( v, i ) => i ) );
		expect( data.nodes.byteLength + data.leaves.byteLength ).toBeLessThan( bvh._roots[ 0 ].byteLength );

	} );

	it( 'splits oversized binary leaves without changing their primitive ranges', () => {

		const positions = [];
		for ( let i = 0; i < 30; i ++ ) {

			positions.push( 0, 0, 0, 1, 0, 0, 0, 1, 0 );

		}

		const geometry = new BufferGeometry();
		geometry.setAttribute( 'position', new Float32BufferAttribute( positions, 3 ) );
		const bvh = new MeshBVH( geometry, { targetLeafSize: 1 } );
		const builder = new CWBVHBuilder();
		const info = builder.add( bvh._roots[ 0 ], 0, ( offset, count ) => ( { value: offset, meta: count } ) );
		const data = builder.build();

		expect( validateAndCollectPrimitives( geometry, data, info.root ).sort( ( a, b ) => a - b ) ).toEqual( Array.from( { length: 30 }, ( v, i ) => i ) );

	} );

	it( 'updates compressed nodes when transforms change', () => {

		const mesh = new Mesh( new TorusKnotGeometry( 1, 0.3, 32, 8 ) );
		const bvhData = new BVHComputeData( mesh, { useCompressedWideBVH: true } );
		bvhData.update();

		const nodes = bvhData.storage.nodes.proxyNode.value.array;
		const leaves = bvhData.storage.leaves.proxyNode.value.array;
		expect( nodes.length % CWBVH_NODE_U32 ).toBe( 0 );
		expect( leaves.length % 2 ).toBe( 0 );

		const before = nodes.slice();
		const beforeLeaves = leaves.slice();
		const tlasNodeLength = bvhData._cwbvhInfo.tlasNodeCount * CWBVH_NODE_U32;
		mesh.position.x = 2;
		mesh.updateMatrixWorld();
		bvhData.updateTransforms();
		expect( nodes ).not.toEqual( before );
		expect( nodes.subarray( tlasNodeLength ) ).toEqual( before.subarray( tlasNodeLength ) );
		expect( leaves ).toEqual( beforeLeaves );

	} );

	it( 'generates the compressed traversal WGSL', () => {

		const mesh = new Mesh( new TorusKnotGeometry( 1, 0.3, 8, 4 ) );
		const bvhData = new BVHComputeData( mesh, { useCompressedWideBVH: true } );
		bvhData.update();

		const computeFn = wgslFn( /* wgsl */`
			fn compute() -> void {

				var ray: Ray;
				ray.origin = vec3f( 0.0 );
				ray.direction = vec3f( 0.0, 0.0, - 1.0 );
				var hit: IntersectionResult;
				bvh_RaycastFirstHit( ray, &hit );

			}
		`, [ bvhData.fns.raycastFirstHit ] );
		const renderer = {
			backend: { isWebGPUBackend: true },
			contextNode: context(),
			coordinateSystem: WebGPUCoordinateSystem,
			getMRT: () => null,
			getOutputRenderTarget: () => null,
			getRenderTarget: () => null,
			hasFeature: () => false,
			nodes: {},
			shadowMap: { enabled: false, type: 0 },
		};
		const builder = new WGSLNodeBuilder( computeFn().computeKernel( [ 1 ] ), renderer );
		builder.build();

		expect( builder.computeShader ).toContain( 'data : array<u32, 20>' );
		expect( builder.computeShader ).toContain( 'info : u32' );
		expect( builder.computeShader ).not.toContain( 'leaf.meta' );
		expect( builder.computeShader ).toContain( 'fn bvh_GetCWBVHChildBounds' );
		expect( builder.computeShader ).toContain( 'var stack: array<u32, 128u>' );

	} );

} );
