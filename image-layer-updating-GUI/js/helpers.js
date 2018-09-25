"use strict";

/*
Convert an array of arrays into a row-major flat buffer.
*/
function flatten_Float32Array( array ) {
    const nrows = array.length;
    const ncols = array[0].length;
    
    var result = new Float32Array( nrows*ncols );
    for( let row = 0; row < nrows; ++row ) {
        for( let col = 0; col < ncols; ++col ) {
            result[row*ncols+col] = array[row][col];
        }
    }
    return result;
}
function flatten_Int32Array( array ) {
    const nrows = array.length;
    const ncols = array[0].length;
    
    var result = new Int32Array( nrows*ncols );
    for( let row = 0; row < nrows; ++row ) {
        for( let col = 0; col < ncols; ++col ) {
            result[row*ncols+col] = array[row][col];
        }
    }
    return result;
}
/*
Expand a row-major flat buffer into an array of arrays with n rows.
*/
function inflate_Float32Array_2D( data, nrows ) {
    let array = new Float32Array( data );
    
    if( array.length % nrows !== 0 ) console.error( "inflate_Float32Array() called but dimensions are impossible." );
    
    var ncols = array.length / nrows;
    var result = Array(nrows);
    for( let row = 0; row < nrows; ++row ) {
        result[row] = Array(ncols);
        for( let col = 0; col < ncols; ++col ) {
            result[row][col] = array[row*ncols+col]
        }
    }
    
    return result;
}
/*
Expand a row-major flat buffer into a length-N array of length-M arrays of length-K arrays.
*/
function inflate_Float32Array_3D( data, N, M, K ) {
    // HACK: Actually we're getting Uint8; convert to Float32.
    // let array = new Float32Array( data );
    let array = new Uint8Array( data );
    
    if( array.length !== N*M*K ) console.error( "inflate_Float32Array() called but dimensions are impossible." );
    
    var result = Array(N);
    for( let row = 0; row < N; ++row ) {
        result[row] = Array(M);
        for( let col = 0; col < M; ++col ) {
            result[row][col] = Array(K);
            for( let channel = 0; channel < K; ++channel ) {
                result[row][col][channel] = array[ (row*M+col)*K + channel ]/255.0;
            }
        }
    }
    
    return result;
}
