from __future__ import print_function
import numpy as np

from collections import Counter
from coordinate_systems import Distance,Angle,Dihedral,OutOfPlane,TranslationX,TranslationY,TranslationZ,RotationA,RotationB,RotationC

class Analyze:
    def find_peaks(self,rtype):
        #rtype 1: growing
        #rtype 2: opting
        #rtype 3: intermediate check
        if rtype==1:
            nnodes=self.nR
        elif rtype==2 or rtype==3:
            nnodes=self.nnodes
        else:
            raise ValueError("find peaks bad input")
        #if rtype==1 or rtype==2:
        #    print "Energy"
        #    print self.energies
        alluptol=0.1
        alluptol2=0.5
        allup=True
        diss=False
        for n in range(1,len(self.energies[:nnodes])):
            if self.energies[n]+alluptol<self.energies[n-1]:
                allup=False
                break

        if self.energies[nnodes-1]>15.0:
            if nnodes-3>0:
                if (abs(self.energies[nnodes-1]-self.energies[nnodes-2])<alluptol2 and
                abs(self.energies[nnodes-2]-self.energies[nnodes-3])<alluptol2 and
                abs(self.energies[nnodes-3]-self.energies[nnodes-4])<alluptol2):
                    print(" possible dissociative profile")
                    diss=True

        print(" nnodes ",nnodes)  
        print(" all uphill? ",allup)
        print(" dissociative? ",diss)
        npeaks1=0
        npeaks2=0
        minnodes=[]
        maxnodes=[]
        if self.energies[1]>self.energies[0]:
            minnodes.append(0)
        if self.energies[nnodes-1]<self.energies[nnodes-2]:
            minnodes.append(nnodes-1)
        for n in range(self.n0,nnodes-1):
            if self.energies[n+1]>self.energies[n]:
                if self.energies[n]<self.energies[n-1]:
                    minnodes.append(n)
            if self.energies[n+1]<self.energies[n]:
                if self.energies[n]>self.energies[n-1]:
                    maxnodes.append(n)

        print(" min nodes ",minnodes)
        print(" max nodes ", maxnodes)
        npeaks1 = len(maxnodes)
        #print "number of peaks is ",npeaks1
        ediff=0.5
        PEAK4_EDIFF = 2.0
        if rtype==1:
            ediff=1.
        if rtype==3:
            ediff=PEAK4_EDIFF

        if rtype==1:
            nmax = np.argmax(self.energies[:self.nR])
            emax = float(max(self.energies[:self.nR]))
        else:
            emax = float(max(self.energies))
            nmax = np.argmax(self.energies)

        print(" emax and nmax in find peaks %3.4f,%i " % (emax,nmax))

        #check if any node after peak is less than 2 kcal below
        for n in maxnodes:
            diffs=( self.energies[n]-e>ediff for e in self.energies[n:nnodes])
            if any(diffs):
                found=n
                npeaks2+=1
        npeaks = npeaks2
        print(" found %i significant peak(s) TOL %3.2f" %(npeaks,ediff))

        #handle dissociative case
        if rtype==3 and npeaks==1:
            nextmin=0
            for n in range(found,nnodes-1):
                if n in minnodes:
                    nextmin=n
                    break
            if nextmin>0:
                npeaks=2

        #if rtype==3:
        #    return nmax
        if allup==True and npeaks==0:
            return -1
        if diss==True and npeaks==0:
            return -2

        return npeaks

    def past_ts(self):
        ispast=ispast1=ispast2=ispast3=0
        THRESH1=5.
        THRESH2=3.
        THRESH3=-1.
        THRESHB=0.05
        CTHRESH=0.005
        OTHRESH=-0.015
        emax = -100.
        nodemax =1
        #n0 is zero until after finished growing
        ns = self.n0-1
        if ns<nodemax: ns=nodemax

        print(" Energies",end=' ')
        for n in range(ns,self.nR):
            print(" {:4.3f}".format(self.energies[n]),end=' ')
            if self.energies[n]>emax:
                nodemax=n
                emax=self.energies[n]
        print("\n nodemax ",nodemax)

        for n in range(nodemax,self.nR):
            if self.energies[n]<emax-THRESH1:
                ispast1+=1
            if self.energies[n]<emax-THRESH2:
                ispast2+=1
            if self.energies[n]<emax-THRESH3:
                ispast3+=1
            if ispast1>1:
                break
        print(" ispast1",ispast1)
        print(" ispast2",ispast2)
        print(" ispast3",ispast3)

        #TODO 5/9/2019 what about multiple constraints
        #cgrad = self.nodes[self.nR-1].gradient[0]
        constraints = self.nodes[self.nR-1].constraints
        gradient = self.nodes[self.nR-1].gradient

        overlap = np.dot(gradient.T,constraints)
        cgrad = overlap*constraints

        cgrad = np.linalg.norm(cgrad)*np.sign(overlap)
        #cgrad = np.sum(cgrad)

        print((" cgrad: %4.3f nodemax: %i nR: %i" %(cgrad,nodemax,self.nR)))


        # 6/17 THIS should check if the last node is high in energy
        if cgrad>CTHRESH and not self.nodes[self.nR-1].PES.lot.do_coupling and nodemax != self.TSnode:
            print(" constraint gradient positive")
            ispast=2
        elif ispast1>0 and cgrad>OTHRESH:
            print(" over the hill(1)")
            ispast=1
        elif ispast2>1:
            print(" over the hill(2)")
            ispast=1
        else:
            ispast=0

        if ispast==0:
            bch=self.check_for_reaction_g(1)
            if ispast3>1 and bch:
                print("over the hill(3) connection changed %r " %bch)
                ispast=3
        print(" ispast=",ispast)
        return ispast

    def check_for_reaction_g(self,rtype):

        c = Counter(elem[0] for elem in self.driving_coords)
        nadds = c['ADD']
        nbreaks = c['BREAK']
        isrxn=False

        if (nadds+nbreaks) <1:
            return False
        nadded=0
        nbroken=0 
        nnR = self.nR-1
        xyz = self.nodes[nnR-1].xyz
        atoms = self.nodes[nnR].atoms

        for i in self.driving_coords:
            if "ADD" in i:
                index = [i[1]-1, i[2]-1]
                bond = Distance(index[0],index[1])
                d = bond.value(xyz)
                d0 = (atoms[index[0]].vdw_radius + atoms[index[1]].vdw_radius)/2
                if d<d0:
                    nadded+=1
            if "BREAK" in i:
                index = [i[1]-1, i[2]-1]
                bond = Distance(index[0],index[1])
                d = bond.value(xyz)
                d0 = (atoms[index[0]].vdw_radius + atoms[index[1]].vdw_radius)/2
                if d>d0:
                    nbroken+=1
        if rtype==1:
            if (nadded+nbroken)>=(nadds+nbreaks): 
                isrxn=True
                #isrxn=nadded+nbroken
        else:
            isrxn=True
            #isrxn=nadded+nbroken
        print(" check_for_reaction_g isrxn: %r nadd+nbrk: %i" %(isrxn,nadds+nbreaks))
        return isrxn

    def check_for_reaction(self):
        isrxn = self.check_for_reaction_g(1)
        minnodes=[]
        maxnodes=[]
        wint=0
        if self.energies[1]>self.energies[0]:
            minnodes.append(0)
        if self.energies[nnodes-1]<self.energies[nnodes-2]:
            minnodes.append(nnodes-1)
        for n in range(self.n0,self.nnodes-1):
            if self.energies[n+1]>self.energies[n]:
                if self.energies[n]<self.energies[n-1]:
                    minnodes.append(n)
            if self.energies[n+1]<self.energies[n]:
                if self.energies[n]>self.energies[n-1]:
                    maxnodes.append(n)
        if len(minnodes)>2 and len(maxnodes)>1:
            wint=minnodes[1] # the real reaction ends at first minimum
            print(" wint ", wint)

        return isrxn,wint


    def calc_grad(self):
        totalgrad = 0.0
        gradrms = 0.0
        sum_gradrms = 0.0
        for i,ico in zip(list(range(1,self.nnodes-1)),self.nodes[1:self.nnodes-1]):
            if ico!=None:
                print(" node: {:2} gradrms: {:1.4}".format(i,float(ico.gradrms)),end='')
                if i%5 == 0:
                    print()
                totalgrad += ico.gradrms*self.rn3m6
                gradrms += ico.gradrms*ico.gradrms
                sum_gradrms += ico.gradrms
        print('')
        #TODO wrong for growth
        gradrms = np.sqrt(gradrms/(self.nnodes-2))
        return totalgrad,gradrms,sum_gradrms


