function fit=BPSO_FITNESS_FUNCTION(Pposition,classnum,totsize)
global classmeanarray totalmeanarray
sum=0;
for i=1:classnum
    diff=double(abs(totalmeanarray(1:totsize)-classmeanarray{i}(1:totsize)));
    tpose=diff';
    diff = diff.*double(Pposition);
    mul=mtimes(diff,tpose);
    sum=double(sum)+double(mul);
end
 fit=sqrt(sum);
end