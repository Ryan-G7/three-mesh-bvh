import { BYTES_PER_NODE, UINT32_PER_NODE } from '../../core/Constants.js';
import { COUNT, IS_LEAF, OFFSET, RIGHT_NODE } from '../../core/utils/nodeBufferUtils.js';

export const CWBVH_NODE_BYTES = 80;
export const CWBVH_NODE_U32 = CWBVH_NODE_BYTES / 4;

const MAX_CHILDREN = 8;
const MAX_LEAF_INDEX = 0x7fffffff;

const _float32 = new Float32Array( 1 );
const _uint32 = new Uint32Array( _float32.buffer );

function nextFloatDown( value ) {

	_float32[ 0 ] = value;
	if ( ! Number.isFinite( value ) ) return value;

	if ( value === 0 ) {

		_uint32[ 0 ] = 0x80000001;

	} else if ( value > 0 ) {

		_uint32[ 0 ] --;

	} else {

		_uint32[ 0 ] ++;

	}

	return _float32[ 0 ];

}

function getBoundsCenter( bounds, axis ) {

	return ( bounds[ axis ] + bounds[ axis + 3 ] ) * 0.5;

}

function getOctant( bounds, parentBounds ) {

	let result = 0;
	for ( let axis = 0; axis < 3; axis ++ ) {

		if ( getBoundsCenter( bounds, axis ) > getBoundsCenter( parentBounds, axis ) ) {

			result |= 1 << ( 2 - axis );

		}

	}

	return result;

}

function bitCount( value ) {

	value = value - ( ( value >> 1 ) & 0x55 );
	value = ( value & 0x33 ) + ( ( value >> 2 ) & 0x33 );
	return ( value + ( value >> 4 ) ) & 0x0f;

}

function assignSlots( children, parentBounds ) {

	const slots = new Array( MAX_CHILDREN ).fill( null );
	for ( let i = 0, l = children.length; i < l; i ++ ) {

		const child = children[ i ];
		const preferred = getOctant( child.bounds, parentBounds );
		let slot = preferred;

		if ( slots[ slot ] !== null ) {

			let bestDistance = Infinity;
			for ( let candidate = 0; candidate < MAX_CHILDREN; candidate ++ ) {

				if ( slots[ candidate ] === null ) {

					const distance = bitCount( candidate ^ preferred );
					if ( distance < bestDistance ) {

						bestDistance = distance;
						slot = candidate;

					}

				}

			}

		}

		slots[ slot ] = child;

	}

	return slots;

}

function getWideChildren( node ) {

	if ( node.isLeaf ) return node.count === 0 ? [] : [ node ];

	const children = [ node.left, node.right ];
	while ( children.length < MAX_CHILDREN ) {

		let candidate = - 1;
		let candidateCount = - 1;
		for ( let i = 0, l = children.length; i < l; i ++ ) {

			const child = children[ i ];
			if ( ! child.isLeaf && child.primitiveCount > candidateCount ) {

				candidate = i;
				candidateCount = child.primitiveCount;

			}

		}

		if ( candidate === - 1 ) break;

		const child = children[ candidate ];
		children.splice( candidate, 1, child.left, child.right );

	}

	return children;

}

function readTree( root, nodeIndex ) {

	const float32 = new Float32Array( root );
	const uint32 = new Uint32Array( root );
	const uint16 = new Uint16Array( root );

	return readNode( nodeIndex );

	function readNode( index ) {

		const index32 = index * UINT32_PER_NODE;
		const index16 = index32 * 2;
		const bounds = Array.from( float32.subarray( index32, index32 + 6 ) );

		if ( IS_LEAF( index16, uint16 ) ) {

			const count = COUNT( index16, uint16 );
			return {
				bounds,
				offset: OFFSET( index32, uint32 ),
				count,
				primitiveCount: count,
				isLeaf: true,
			};

		}

		const left = readNode( index + 1 );
		const right = readNode( RIGHT_NODE( index32, uint32 ) / UINT32_PER_NODE );
		return {
			bounds,
			left,
			right,
			primitiveCount: left.primitiveCount + right.primitiveCount,
			isLeaf: false,
		};

	}

}

function getQuantization( bounds ) {

	const origin = new Array( 3 );
	const exponent = new Array( 3 );
	const scale = new Array( 3 );

	for ( let axis = 0; axis < 3; axis ++ ) {

		const min = bounds[ axis ];
		const max = bounds[ axis + 3 ];
		origin[ axis ] = nextFloatDown( min );

		let axisExponent = - 126;
		if ( max > origin[ axis ] ) {

			axisExponent = Math.ceil( Math.log2( ( max - origin[ axis ] ) / 255 ) );

		}

		axisExponent = Math.max( - 126, Math.min( 127, axisExponent ) );
		let axisScale = Math.pow( 2, axisExponent );
		while ( Math.fround( origin[ axis ] + 255 * axisScale ) < max && axisExponent < 127 ) {

			axisExponent ++;
			axisScale *= 2;

		}

		if ( Math.fround( origin[ axis ] + 255 * axisScale ) < max ) {

			throw new Error( 'CWBVHBuilder: Bounds cannot be represented with 8-bit quantization.' );

		}

		exponent[ axis ] = axisExponent;
		scale[ axis ] = axisScale;

	}

	return { origin, exponent, scale };

}

function quantizeMin( value, origin, scale ) {

	let result = Math.max( 0, Math.min( 255, Math.floor( ( value - origin ) / scale ) ) );
	while ( result > 0 && Math.fround( origin + result * scale ) > value ) result --;
	return result;

}

function quantizeMax( value, origin, scale ) {

	let result = Math.max( 0, Math.min( 255, Math.ceil( ( value - origin ) / scale ) ) );
	while ( result < 255 && Math.fround( origin + result * scale ) < value ) result ++;
	return result;

}

function packBytes( target, offset, values ) {

	for ( let word = 0; word < 2; word ++ ) {

		let value = 0;
		for ( let byte = 0; byte < 4; byte ++ ) {

			value |= values[ word * 4 + byte ] << ( byte * 8 );

		}

		target[ offset + word ] = value >>> 0;

	}

}

function getMaxStackSize( nodes, nodeIndex ) {

	const slots = nodes[ nodeIndex ].slots.filter( slot => slot !== null );
	let result = Math.max( 1, slots.length );
	for ( let i = 0, l = slots.length; i < l; i ++ ) {

		const slot = slots[ i ];
		if ( slot.nodeIndex !== undefined ) {

			result = Math.max( result, slots.length - 1 + getMaxStackSize( nodes, slot.nodeIndex ) );

		}

	}

	return result;

}

/**
 * Converts packed binary BVH roots into 8-way, quantized CWBVH nodes. Multiple trees can be
 * appended so TLAS leaves can reference BLAS root indices in the same node array.
 *
 * The 80-byte layout follows the compressed-wide design described by Ylitie et al. while keeping
 * leaf ranges in a compact side buffer so existing MeshBVH leaf sizes remain supported.
 * @see https://research.nvidia.com/publication/2017-07_efficient-incoherent-ray-traversal-gpus-through-compressed-wide-bvhs
 */
export class CWBVHBuilder {

	constructor() {

		this.nodes = [];
		this.leaves = [];

	}

	/**
	 * @param {ArrayBuffer|SharedArrayBuffer} root
	 * @param {number} nodeIndex
	 * @param {(offset:number,count:number) => {value:number|(() => number), meta:number}} getLeaf
	 * @returns {{root:number,nodeCount:number,maxStackSize:number}}
	 */
	add( root, nodeIndex, getLeaf ) {

		if ( root.byteLength % BYTES_PER_NODE !== 0 ) {

			throw new Error( 'CWBVHBuilder: Invalid packed BVH buffer.' );

		}

		const tree = readTree( root, nodeIndex );
		const start = this.nodes.length;
		this.nodes.push( null );
		this._writeNode( tree, start, getLeaf );

		return {
			root: start,
			nodeCount: this.nodes.length - start,
			maxStackSize: getMaxStackSize( this.nodes, start ),
		};

	}

	build() {

		// 3 float origins, 3 signed byte exponents, child / leaf bases, 6 x 8 byte
		// quantized bounds arrays, and 8 metadata bytes.
		const nodeBuffer = new ArrayBuffer( this.nodes.length * CWBVH_NODE_BYTES );
		const nodeF32 = new Float32Array( nodeBuffer );
		const nodeU32 = new Uint32Array( nodeBuffer );

		for ( let i = 0, l = this.nodes.length; i < l; i ++ ) {

			const node = this.nodes[ i ];
			const index = i * CWBVH_NODE_U32;
			for ( let axis = 0; axis < 3; axis ++ ) nodeF32[ index + axis ] = node.origin[ axis ];

			nodeU32[ index + 3 ] =
				( node.exponent[ 0 ] & 0xff ) |
				( ( node.exponent[ 1 ] & 0xff ) << 8 ) |
				( ( node.exponent[ 2 ] & 0xff ) << 16 );
			nodeU32[ index + 4 ] = node.childBase;
			nodeU32[ index + 5 ] = node.leafBase;

			for ( let bound = 0; bound < 6; bound ++ ) {

				packBytes( nodeU32, index + 6 + bound * 2, node.quantizedBounds[ bound ] );

			}

			packBytes( nodeU32, index + 18, node.metadata );

		}

		const leafBuffer = new Uint32Array( this.leaves.length * 2 );
		for ( let i = 0, l = this.leaves.length; i < l; i ++ ) {

			const leaf = this.leaves[ i ];
			const value = typeof leaf.value === 'function' ? leaf.value() : leaf.value;
			if ( ! Number.isInteger( value ) || value < 0 || value > 0xffffffff ) {

				throw new Error( `CWBVHBuilder: Leaf value ${ value } cannot be represented as uint32.` );

			}

			leafBuffer[ i * 2 ] = value;
			leafBuffer[ i * 2 + 1 ] = leaf.meta;

		}

		return { nodes: new Uint32Array( nodeBuffer ), leaves: leafBuffer };

	}

	_writeNode( tree, nodeIndex, getLeaf ) {

		const children = getWideChildren( tree );
		const slots = assignSlots( children, tree.bounds );
		const internalSlots = slots.filter( child => child !== null && ! child.isLeaf );
		const childBase = internalSlots.length === 0 ? 0 : this.nodes.length;
		for ( let i = 0, l = internalSlots.length; i < l; i ++ ) this.nodes.push( null );

		const leafBase = this.leaves.length;
		const metadata = new Array( MAX_CHILDREN ).fill( 0 );
		let leafOffset = 0;
		let internalOffset = 0;

		for ( let slot = 0; slot < MAX_CHILDREN; slot ++ ) {

			const child = slots[ slot ];
			if ( child === null ) continue;

			if ( child.isLeaf ) {

				metadata[ slot ] = 0x40 | leafOffset;
				this.leaves.push( getLeaf( child.offset, child.count ) );
				leafOffset ++;

			} else {

				child.nodeIndex = childBase + internalOffset;
				metadata[ slot ] = 0x80 | internalOffset;
				internalOffset ++;

			}

		}

		if ( leafOffset > MAX_CHILDREN || this.leaves.length > MAX_LEAF_INDEX ) {

			throw new Error( 'CWBVHBuilder: Leaf index exceeds the compressed node encoding.' );

		}

		const { origin, exponent, scale } = getQuantization( tree.bounds );
		const quantizedBounds = Array.from( { length: 6 }, () => new Array( MAX_CHILDREN ).fill( 0 ) );
		for ( let slot = 0; slot < MAX_CHILDREN; slot ++ ) {

			const child = slots[ slot ];
			if ( child === null ) continue;

			for ( let axis = 0; axis < 3; axis ++ ) {

				quantizedBounds[ axis * 2 ][ slot ] = quantizeMin( child.bounds[ axis ], origin[ axis ], scale[ axis ] );
				quantizedBounds[ axis * 2 + 1 ][ slot ] = quantizeMax( child.bounds[ axis + 3 ], origin[ axis ], scale[ axis ] );

			}

		}

		this.nodes[ nodeIndex ] = {
			origin,
			exponent,
			childBase,
			leafBase,
			quantizedBounds,
			metadata,
			slots,
		};

		for ( let slot = 0; slot < MAX_CHILDREN; slot ++ ) {

			const child = slots[ slot ];
			if ( child !== null && ! child.isLeaf ) {

				this._writeNode( child, child.nodeIndex, getLeaf );

			}

		}

	}

}
