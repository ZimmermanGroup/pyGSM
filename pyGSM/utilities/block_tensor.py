import numpy as np
from scipy.linalg import block_diag
from .nifty import printcool,pvec1d
import sys
from .math_utils import orthogonalize,conjugate_orthogonalize


class block_tensor(object):

    def __init__(self,matlist,cnorms=None):
        self.matlist = matlist
        if cnorms is None:
            cnorms = np.zeros((self.shape[1],1))
        self.cnorms=cnorms

    def __repr__(self):
        lines= [" block matrix: # blocks = {}".format(self.num_blocks)]
        count=0
        for m in self.matlist:
            lines.append(str(m))
            count+=1
            if count>10:
                print('truncating printout')
                break
        return '\n'.join(lines)

    @staticmethod
    def full_matrix(A):
        return block_diag(*A.matlist)

    @property
    def num_blocks(self):
        return len(self.matlist)

    
    @staticmethod
    def zeros_like(BM):
        return block_tensor( [ np.zeros_like(A) for A in BM.matlist ] ) 
    
    def __add__(self,rhs):
        print("adding")
        if isinstance(rhs, self.__class__):
            print("adding block matrices!")
            assert(self.shape == rhs.shape)
            return block_tensor( [A+B for A,B in zip(self.matlist,rhs.matlist) ] )
        elif isinstance(rhs,float) or isinstance(rhs,int):
            return block_tensor( [A+rhs for A in self.matlist ])
        else: 
            raise NotImplementedError

    def __radd__(self,lhs):
        return self.__add__(lhs)

    def __mul__(self,rhs):
        if isinstance(rhs, self.__class__):
            assert(self.shape == rhs.shape)
            return block_tensor( [A*B for A,B in zip(self.matlist,rhs.matlist)] )
        elif isinstance(rhs,float) or isinstance(rhs,int):
            return block_tensor( [A*rhs for A in self.matlist ])
        else: 
            raise NotImplementedError

    def __rmul__(self,lhs):
        return self.__mul__(lhs)

    def __len__(self):  #size along first axis
        return np.sum([len(A) for A in self.matlist])

    def __truediv__(self,rhs):
        if isinstance(rhs, self.__class__):
            assert(self.shape == rhs.shape)
            return block_tensor( [A/B for A,B in zip(self.matlist,rhs.matlist)] )
        elif isinstance(rhs,float) or isinstance(rhs,int):
            return block_tensor( [A/rhs for A in self.matlist ])
        elif isinstance(rhs,np.ndarray):
            answer = []
            s=0
            for block in self.matlist:
                e=block.shape[1]+s
                answer.append(block/rhs[s:e])
                s=e
            return block_tensor(answer)
        else: 
            raise NotImplementedError


    @property
    def shape(self):
        tot = (0,0,0)
        for a in self.matlist:
            tot = tuple(map(sum,zip(a.shape,tot)))
        return tot

    @staticmethod
    def transpose(A):
        return block_tensor( [ A.T for A in A.matlist] )

    @staticmethod
    def dot(left,right):
        def block_vec_dot(block,vec):
            if vec.ndim==2 and vec.shape[1]==1:
                vec = vec.flatten()
            #if block.cnorms is None:
            s=0
            result=[]
            for A in block.matlist:
                e = s + np.shape(A)[1]
                result.append(np.dot(A,vec[s:e]))
                s=e
            return np.reshape(np.concatenate(result),(-1,1))
        def vec_block_dot(vec,block,**kwargs):
            if vec.ndim==2 and vec.shape[1]==1:
                vec = vec.flatten()
            #if block.cnorms is None:
            s=0
            result=[]
            for A in block.matlist:
                e = s + np.shape(A)[1]
                result.append(np.dot(vec[s:e],A))
                s=e
            return np.reshape(np.concatenate(result),(-1,1))

        # (1) both are block matrices
        if isinstance(left,block_tensor) and isinstance(right,block_tensor):
            return block_tensor([np.dot(A,B) for A,B in zip(left.matlist,right.matlist)])
        # (2) left is np.ndarray with a vector shape
        elif isinstance(left,np.ndarray) and (left.ndim==1 or left.shape[1]==1) and isinstance(right,block_tensor):
            return vec_block_dot(left,right)
        # (3) right is np.ndarray with a vector shape
        elif isinstance(right,np.ndarray) and (right.ndim==1 or right.shape[1]==1) and isinstance(left,block_tensor):
            return block_vec_dot(left,right)
        # (4) l/r is a matrix
        elif isinstance(left,np.ndarray) and left.ndim==2: 
            #           
            # [ A | B ] [ C 0 ] = [ AC BD ]
            #           [ 0 D ]
            sc=0
            tmp_ans=[]
            for A in right.matlist:
                ec = sc+A.shape[0]
                tmp_ans.append(np.dot(left[:,sc:ec],A))
                sc=ec
            dot_product=np.hstack(tmp_ans)
            return dot_product

        elif isinstance(right,np.ndarray) and right.ndim==2:
            #           
            # [ A | 0 ] [ C ] = [ AC ]
            # [ 0 | B ] [ D ]   [ BD ]
            sc=0
            tmp_ans=[]
            for A in left.matlist:
                ec = sc+A.shape[1]
                tmp_ans.append(np.dot(A,right[sc:ec,:]))
                sc=ec
            dot_product=np.vstack(tmp_ans)
            return dot_product
        else: 
            raise NotImplementedError


#if __name__=="__main__":

#A = [np.array([[1,2],[3,4]]), np.array([[5,6],[7,8]])]
#B = [np.array([[1,2],[3,4]]), np.array([[5,6],[7,8]])]
#Ab = bm(A)
#Bb = bm(B)
#
#print("A")
#print(Ab)
#
#print("B")
#print(Bb)
#
## test 1
#print("test 1 adding block matrices")
#Cb = Ab+Bb
#print(Cb)
#
#print("test 2 adding block matrix and float")
#Db = Ab+2
#print(Db)
#
#print("test 3 reversing order of addition")
#Eb = 2+Ab
#print(Eb)
#
#print("test 4 block multiplication")
#Fb = Ab*Bb
#print(Fb)
#
#print("test 5 block multiplication by scalar")
#Gb = Ab*2
#print(Gb)
#
#print("test 6 reverse block mult by scalar")
#Hb = 2*Ab
#print(Hb)
#
#print("test 7 total len")
#print(len(Hb))
#
#print("test 8 shape")
#print(Hb.shape)
#
#print("test dot product with block matrix")
#Ib = bm.dot(Ab,Bb)
#print(Ib)
#
#print("test dot product with np vector")
#Jb = bm.dot(Ab,np.array([1,2,3,4]))
#print(Jb)
#
#print("Test dot product with np 2d vector shape= (x,1)")
#a = np.array([[1,2,3,4]]).T
#Kb = bm.dot(Ab,a)
#print(Kb)
#
#print("test dot product with non-block array")
#fullmat = np.random.randint(5,size=(4,4))
#print(" full mat to mult")
#print(fullmat)
#A = [np.array([[1,2,3],[4,5,6]]), np.array([[7,8,9],[10,11,12]])]
#Ab = bm(A)
#print(" Ab")
#print(bm.full_matrix(Ab))
#print('result')
#Mb = np.dot(fullmat,bm.full_matrix(Ab))
#print(Mb)
#Lb = bm.dot(fullmat,Ab)
#print('result of dot product with full mat')
#print(Lb)
#print(Lb == Mb)
#
#print("test dot product with non-block array")
#print(" full mat to mult")
#print(fullmat)
#print(" Ab")
#print(bm.full_matrix(Ab))
#print('result')
#A = [ np.array([[1,2],[3,4],[5,6]]),np.array([[7,8],[9,10],[11,12]])]
#Ab = bm(A)
#print(Ab.shape)
#print(fullmat.shape)
#Mb = np.dot(bm.full_matrix(Ab),fullmat)
#print(Mb)
#Lb = bm.dot(Ab,fullmat)
#print('result of dot product with full mat')
#print(Lb)
#print(Lb == Mb)
#
