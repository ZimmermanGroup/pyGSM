import numpy as np
from scipy.linalg import block_diag
from .nifty import printcool,pvec1d
import sys
from .math_utils import orthogonalize,conjugate_orthogonalize


class block_matrix(object):

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

    # IDEA: everywhere a dot product of DLC is done, use the conjugate 
    # dot product, also use the conjugate_orthogonalize to orthogonalize
    @staticmethod
    def project_conjugate_constraint(BM,constraints,G):
        def ov(vi, vj):
            return np.linalg.multi_dot([vi, G, vj])

        # the constraints need to be orthonormalized on G
        constraints = conjugate_orthogonalize(constraints,G)

        # (a) need to zero some segments (corresponding to the blocks of Vecs) of the constraints if their magnitude is small
        s=0
        for block in BM.matlist:
            size=len(block)
            e=s+size
            for constraint in constraints.T:
                if (constraint[s:e]==0.).all():
                    pass
                elif np.linalg.norm(constraint[s:e])<1.0e-3:
                    constraint[s:e] = np.zeros(size)
            s=e

        # (b) renormalizing the constraints on the surface G
        norms = np.sqrt((ov(constraints.T,constraints).sum(axis=0,keepdims=True)))
        constraints = constraints/norms
        #print('constraints after renormalizing')
        #print(constraints.T)

        # (c) need to save the magnitude of the constraints in each segment since they 
        # will be renormalized for each block
        cnorms = np.zeros((BM.shape[1],constraints.shape[1]))
        sr=0
        sc=0
        newblocks=[]
        for block in BM.matlist:
            size_r=block.shape[0]
            size_c=block.shape[1]
            er=sr+size_r
            ec=sc+size_c
            flag=False
            tmpc = []
            for count,constraint in enumerate(constraints.T):

                # CRA 81219 what to do here? mag of real or g-space?
                #mag = np.linalg.norm(constraint[sr:er])
                mag = np.sqrt(np.linalg.multi_dot([constraint[sr:er],G[sr:er,sr:er],constraint[sr:er]]))

                # concatenating the block to each constraint if the constraint is greater than parameter
                if mag>1.e-3: 
                    cnorms[sc+count,count]=mag
                    tmpc.append(constraint[sr:er]/mag)
                    flag=True
            if flag:
                tmpc = np.asarray(tmpc).T
                if len(tmpc)!=len(block):
                    raise RuntimeError
                newblocks.append(np.hstack((tmpc,block)))
            else:
                newblocks.append(block)
            sr=er
            sc=ec

        # (d) orthogonalize each block
        ans=[]
        sr=0
        sc=0
        count=0
        for nb,ob in zip(newblocks,BM.matlist):
            size_c=ob.shape[1]
            size_r=block.shape[0]
            er=sr+size_r
            ec=sc+size_c
            num_c=0
            flag=False
            for c in cnorms.T:
                if any(c[sc:ec]!=0.):
                    num_c +=1
                    flag=True
            if flag:
                ans.append(conjugate_orthogonalize(nb,G[sr:er,sr:er],num_c))
            else:
                ans.append(conjugate_orthogonalize(nb,G[sr:er,sr:er]))
                #ans.append(ob)
            sc=ec
            sr=er
            count+=1
        return block_matrix(ans,cnorms)

    #TODO 8/10/2019 write a detailed explanation for this method
    @staticmethod
    def project_constraint(BM,constraints):
        assert( len(constraints) == len(BM) )

        # (a) need to zero some segments (corresponding to the blocks of Vecs) of the constraints if their magnitude is small
        s=0
        for block in BM.matlist:
            size=len(block)
            e=s+size
            for constraint in constraints.T:
                if (constraint[s:e]==0.).all():
                    pass
                elif np.linalg.norm(constraint[s:e])<1.0e-2:
                    constraint[s:e] = np.zeros(size)
            s=e

        # (b) renormalizing the constraints
        norms = np.sqrt((constraints*constraints).sum(axis=0,keepdims=True))
        #print('norms')
        #print(norms)
        constraints = constraints/norms
        
        # (c) need to save the magnitude of the constraints in each segment since they 
        # will be renormalized for each block
        cnorms = np.zeros((BM.shape[1],constraints.shape[1]))
        sr=0
        sc=0
        newblocks=[]
        for block in BM.matlist:
            size_r=block.shape[0]
            size_c=block.shape[1]
            er=sr+size_r
            ec=sc+size_c
            flag=False
            tmpc = []
            for count,constraint in enumerate(constraints.T):
                mag = np.linalg.norm(constraint[sr:er])
                # (d) concatenating the block to each constraint if the constraint is greater than parameter
                if mag>1.e-2: 
                    cnorms[sc+count,count]=mag
                    tmpc.append(constraint[sr:er]/mag)
                    flag=True
            if flag:
                tmpc = np.asarray(tmpc).T
                if len(tmpc)!=len(block):
                    raise RuntimeError
                newblocks.append(np.hstack((tmpc,block)))
            else:
                newblocks.append(block)
            sr=er
            sc=ec

        #sc=0
        #for i,block in enumerate(BM.matlist):
        #    size_c=block.shape[1]
        #    ec=sc+size_c
        #    for c in cnorms.T:
        #        if any(c[sc:ec]!=0.):
        #            #print(c[np.nonzero(c[sc:ec])[0]])
        #            print(c[sc:ec])
        #    sc=ec

        # TMP print out
        #print(" printing cnorms")
        #np.savetxt('cnorms.txt',cnorms.T)
        #sc=0
        #only_tan=[]
        #only_dg=[]
        #only_dv=[]
        #if cnorms.shape[1]==3:
        #    for b in BM.matlist:
        #        ec=sc+b.shape[1]

        #        #HACK FOR dpb
        #        for i in range(30):
        #            only_tan.append(cnorms[sc,0])
        #            only_dg.append(cnorms[sc+1,1])
        #            only_dv.append(cnorms[sc+2,2])
        #        sc=ec
        #    vals = np.hstack((only_tan,only_dg,only_dv))
        #    np.savetxt('vals.txt',vals,fmt='%1.2f')

        #print(cnorms.T)
        #check = np.sqrt((cnorms*cnorms).sum(axis=0,keepdims=True))
        #print(" Check normality of cnorms")
        #print(check)
        #print("done")

        #print(" printing out blocks")
        #sc=0
        #count=0
        #for b in BM.matlist:
        #    ec = sc+b.shape[1]
        #    for c in cnorms.T:
        #        if any(c[sc:ec]!=0.):
        #            print('block %d mag %.4f' %(count,np.linalg.norm(c[sc:ec])))
        #            print(c[sc:ec])
        #        else:
        #            print('block %d mag %.4f' %(count,np.linalg.norm(c[sc:ec])))
        #    sc=ec
        #    count+=1
        #print(" done")
       
        assert len(newblocks) == len(BM.matlist), "not proper lengths for zipping"

        #print(" len of nb = {}".format(len(newblocks)))
        #print(" len of ob = {}".format(len(BM.matlist)))
        #count=0
        #for nb,ob in zip(newblocks,BM.matlist):
        #    print(count)
        #    print(nb.shape)
        #    print(ob.shape)
        #    count+=1

        # NEW
        # orthogonalize each sub block
        #print(" Beginning to orthogonalize each sub block")
        ans=[]
        sc=0
        count=0
        for nb,ob in zip(newblocks,BM.matlist):
            #print("On block %d" % count)
            size_c=ob.shape[1]
            ec=sc+size_c
            num_c=0
            flag=False
            for c in cnorms.T:
                #if (c[sc:ec]!=0.).any():
                if any(c[sc:ec]!=0.):
                    num_c +=1
                    #print('block %d mag %.4f' %(count,np.linalg.norm(c[sc:ec])))
                    #print(c[sc:ec])
                    #print('num_c=%d' %num_c)
                    flag=True
            #print(flag)
            if flag:
                #print(" orthogonalizing sublock {} with {} constraints".format(count,num_c))
                #print(ob.shape)
                #print(nb.shape)
                try:
                    a = orthogonalize(nb,num_c)
                    #print("result {}".format(a.shape))
                except:
                    print(" what is happening")
                    print("nb")
                    print(nb)
                    print(nb.shape)
                    print(num_c)
                    print("ob")
                    print(ob)
                ans.append(a)
                #ans.append(orthogonalize(nb,num_c))
            else:
                #print(" appending old block without constraints")
                ans.append(ob)
            sc=ec
            count+=1

            
        return block_matrix(ans,cnorms)


        # (d) concatenating the block to each constraint if the constraint is non-zero
        #sr=0
        #newblocks=[]
        #for block in BM.matlist:
        #    size_r=block.shape[0]
        #    er=sr+size_r
        #    flag=False
        #    tmpc = []
        #    for constraint in constraints.T:
        #        #if (constraint[s:e]!=0.).all():
        #        mag = np.linalg.norm(constraint[sr:er])
        #        if mag>0.:
        #            tmpc.append(constraint[sr:er]/mag)
        #            flag=True
        #    if flag==True:
        #        tmpc = np.asarray(tmpc).T
        #        if len(tmpc)!=len(block):
        #            #print(tmpc.shape)
        #            #print(block.shape)
        #            #print('start %i end %i' %(s,e))
        #            raise RuntimeError
        #        newblocks.append(np.hstack((tmpc,block)))
        #    else:
        #        newblocks.append(block)
        #    sr=er

        #print('cnorms')
        #print(cnorms[np.nonzero(cnorms)[0]])
        #return block_matrix(newblocks,cnorms)


    @staticmethod
    def qr(BM): #only return the Q part
        #print("before qr")
        #print(BM)
        ans = []
        for A in BM.matlist:
            Q,R = np.linalg.qr(A)
            indep = np.where(np.abs(R.diagonal()) >  min_tol)[0]
            ans.append(Q[:,indep])
            if len(indep)>A.shape[1]:
                print(" the basis dimensions are too large.")
                raise RuntimeError
            #tmp = np.dot(Q,R)
            #print(tmp.shape)
            #print("r,q shape")
            #print(R.shape)
            #pvec1d(R[-1,:])
            #ans.append(Q[:,:BM.shape[1]-BM.cnorms.shape[1]])
            #m=A.shape[1]
            ##print(R)
            #for i in range(BM.cnorms.shape[1]):
            #    if np.linalg.norm(R[-1,:])<1.e-3:
            #        m-=1
            #ans.append(Q[:,:m])
        return block_matrix(ans,BM.cnorms)
        #return block_matrix( [ np.linalg.qr(A)[0] for A in BM.matlist ], BM.cnorms)

    @staticmethod
    def diagonal(BM):
        la = [ np.diagonal(A) for A in BM.matlist ]
        return np.concatenate(la)
    
    @staticmethod
    def gram_schmidt(BM):
        ans=[]
        sc=0
        for i,block in enumerate(BM.matlist):
            size_c=block.shape[1]
            ec=sc+size_c
            num_c=0
            for c in BM.cnorms.T:
                if any(c[sc:ec]!=0.):
                    num_c +=1
                    print('block %d mag %.4f' %(i,np.linalg.norm(c[sc:ec])))
                    print(c[sc:ec])
                    print('num_c=%d' %num_c)
            ans.append(orthogonalize(block,num_c))
            sc=ec
        return block_matrix(ans,BM.cnorms)

    @staticmethod
    def eigh(BM):
        eigenvalues=[]
        eigenvectors=[]
        for block in BM.matlist:
            e,v = np.linalg.eigh(block)
            eigenvalues.append(e)
            eigenvectors.append(v)
        return np.concatenate(eigenvalues),block_matrix(eigenvectors)

    @staticmethod
    def zeros_like(BM):
        return block_matrix( [ np.zeros_like(A) for A in BM.matlist ] ) 

    
    def __add__(self,rhs):
        print("adding")
        if isinstance(rhs, self.__class__):
            print("adding block matrices!")
            assert(self.shape == rhs.shape)
            return block_matrix( [A+B for A,B in zip(self.matlist,rhs.matlist) ] )
        elif isinstance(rhs,float) or isinstance(rhs,int):
            return block_matrix( [A+rhs for A in self.matlist ])
        else: 
            raise NotImplementedError

    def __radd__(self,lhs):
        return self.__add__(lhs)

    def __mul__(self,rhs):
        if isinstance(rhs, self.__class__):
            assert(self.shape == rhs.shape)
            return block_matrix( [A*B for A,B in zip(self.matlist,rhs.matlist)] )
        elif isinstance(rhs,float) or isinstance(rhs,int):
            return block_matrix( [A*rhs for A in self.matlist ])
        else: 
            raise NotImplementedError

    def __rmul__(self,lhs):
        return self.__mul__(lhs)

    def __len__(self):  #size along first axis
        return np.sum([len(A) for A in self.matlist])

    def __truediv__(self,rhs):
        if isinstance(rhs, self.__class__):
            assert(self.shape == rhs.shape)
            return block_matrix( [A/B for A,B in zip(self.matlist,rhs.matlist)] )
        elif isinstance(rhs,float) or isinstance(rhs,int):
            return block_matrix( [A/rhs for A in self.matlist ])
        elif isinstance(rhs,np.ndarray):
            answer = []
            s=0
            for block in self.matlist:
                e=block.shape[1]+s
                answer.append(block/rhs[s:e])
                s=e
            return block_matrix(answer)
        else: 
            raise NotImplementedError


    @property
    def shape(self):
        tot = (0,0)
        for a in self.matlist:
            tot = tuple(map(sum,zip(a.shape,tot)))
        return tot

    @staticmethod
    def transpose(A):
        return block_matrix( [ A.T for A in A.matlist] )

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
        if isinstance(left,block_matrix) and isinstance(right,block_matrix):
            return block_matrix([np.dot(A,B) for A,B in zip(left.matlist,right.matlist)])
        # (2) left is np.ndarray with a vector shape
        elif isinstance(left,np.ndarray) and (left.ndim==1 or left.shape[1]==1) and isinstance(right,block_matrix):
            return vec_block_dot(left,right)
        # (3) right is np.ndarray with a vector shape
        elif isinstance(right,np.ndarray) and (right.ndim==1 or right.shape[1]==1) and isinstance(left,block_matrix):
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
