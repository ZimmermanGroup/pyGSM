basis=6-31G*

{multi
direct,
closed,6
occ,9
wf,16,1,0
state,2
CPMCSCF,GRAD,1.1,record=5100.1
CPMCSCF,GRAD,2.1,record=5101.1
}


FORCE;SAMC,5100.1;varsav
FORCE;SAMC,5101.1;varsav
