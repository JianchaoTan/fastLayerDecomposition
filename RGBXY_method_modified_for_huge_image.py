import numpy as np
import time
import scipy
from scipy.spatial import ConvexHull, Delaunay
import scipy.sparse
from numpy import *

def recover_ASAP_weights_using_scipy_delaunay(Hull_vertices, data):
    ############## copy from https://codereview.stackexchange.com/questions/41024/faster-computation-of-barycentric-coordinates-for-many-points (Gareth Rees)
    
    start=time.time()
    # Compute Delaunay triangulation of points.
    tri = Delaunay(Hull_vertices)
    
    end=time.time()

    print "delaunay time: ", end-start

    CHUNK_SIZE = 1000
    
    for i in range(len(data)/CHUNK_SIZE):
        if i%1000==0:
            print i

        end1=time.time()

        targets = data[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE]

        # Find the tetrahedron containing each target (or -1 if not found)
        tetrahedra = tri.find_simplex(targets, tol=1e-6)
    #     print tetrahedra[tetrahedra==-1]

        # Affine transformation for tetrahedron containing each target
        X = tri.transform[tetrahedra, :data.shape[1]]

        # Offset of each target from the origin of its containing tetrahedron
        Y = targets - tri.transform[tetrahedra, data.shape[1]]

        # First three barycentric coordinates of each target in its tetrahedron.
        # The fourth coordinate would be 1 - b.sum(axis=1), but we don't need it.
        b = np.einsum('...jk,...k->...j', X, Y)
        barycoords=np.c_[b,1-b.sum(axis=1)]

    
        end2=time.time()
        
        rows = np.repeat(np.arange(len(targets)).reshape((-1,1)), len(tri.simplices[0]), 1).ravel()
        cols=tri.simplices[tetrahedra].ravel()
        vals = barycoords.ravel()
        weights_list = scipy.sparse.coo_matrix( ( vals, ( rows, cols ) ), shape = ( len(targets), len(Hull_vertices)) ).tocsr()

        end3=time.time()
        
        # print end2-end1, end3-end2
