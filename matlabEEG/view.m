% da = load('/Users/yiranzhao/Dropbox/EEG/work/syncdir/data/tmpdata-1508-7.txt');
da0 = load('/Users/yiranzhao/Dropbox/EEG/work/matlabEEG/aa-train-100.txt');

% da0 = load('/Users/yiranzhao/Dropbox/EEG/work/bci4d2a/data22/data-750-0-1 copy.txt');
% da1 = load('/Users/yiranzhao/Dropbox/EEG/work/bci4d2a/data22/data-750-1-1 copy.txt');
da1=da0;

%tmpdata-3044-3-fft
%tmpdata-1508-7-fft

%%
nch = size(da0,2)-1

t0=1;
lb = da0(t0,nch+1)

t=t0+1;
while t<=size(da0,1) && da0(t,nch+1)==lb && t<=size(da1,1) && da1(t,nch+1)==lb
    t=t+1;
end
t=t-1;

ch =1
x0=da0(t0:1:t-2,ch);
x1=da1(t0:1:t-2,ch);
figure
hold on
plot(x0,'k')
plot(x1,'b')

x = zeros(1,size(x0,1)+size(x1,1));
for i =1:min(size(x0,1),size(x1,1))
    x(2*i-1)=x0(i);x(2*i)=x1(i);
end

figure
plot(x)


%%
