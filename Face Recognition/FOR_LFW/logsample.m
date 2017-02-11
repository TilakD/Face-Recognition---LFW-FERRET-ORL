function logarr = logsample(a)

xc=size(a,2)/2;
yc=size(a,1)/2;
nr=min(size(a,2),size(a,1));
rmin=0.1;
rmax=min(xc,yc);
nw=-2*pi*(nr-1) / log(rmin/rmax);
rmax=min([xc-0.5,yc-0.5]);

%---------------------------------------------------------------------------
t = logtform(rmin, rmax, nr, nw);
nr = t.tdata.nr;        % Get computed values, in case default used
nw = t.tdata.nw;
[U, V, ~] = size(a);
Udata = [1, V] - xc;
Vdata = [1, U] - yc;
Xdata = [0, nr-1];
Ydata = [0, nw-1];
Size = [round(nw), round(nr)];
logarr = imtransform(a, t, ...
    'Udata', Udata, 'Vdata', Vdata, ...
    'Xdata', Xdata, 'Ydata', Ydata, 'Size', Size);
%arr= logsampback(logarr, rmin, rmax);
%figure,imshow(arr);
%fprintf('entropy of reobtained image is %d',entropy(arr));
end