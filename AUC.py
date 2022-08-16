def AUC(negs,poss):
    mini=min(min(negs),min(poss))
    maxi=max(max(negs),max(poss))
    mini=mini-0.00001
    maxi=maxi+0.00001
    delta=(maxi-mini)/100
    FPRs=[]
    TPRs=[]
    for i in range(101):
        tau=mini+delta*i
        FPc=0
        for t in negs:
            if t>=tau:
                FPc=FPc+1
        TPc=0
        for t in poss:
            if t>=tau:
                TPc=TPc+1
        FPR=FPc/len(negs)
        TPR=TPc/len(poss)
        FPRs.append(FPR)
        TPRs.append(TPR)
    auc=0
    for i in range(100):
        auc=auc+(FPRs[i]-FPRs[i+1])*(TPRs[i]+TPRs[i+1])/2
    return auc
