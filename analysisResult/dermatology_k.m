clear on;
close all;
x=1:10;
% 0.7
% KMeans =   [0.72,0.632,0.95,0.674,0.619,1.131,1.052,0.695,0.723,0.829];
% KMeansPlus =  [0.61,0.536,1.09,0.581,0.492,0.838,0.523,0.553,0.597,0.485];
% KMeansM = [0.181,0.115,0.18,0.172,0.118,0.116,0.119,0.115,0.117,0.115];
% 0.8
%KMeans =   [0.973,1.227,0.795,0.735,0.584,0.618,0.865,0.504,0.711,0.772];
%KMeansPlus =  [0.548,0.689,0.590,0.666,0.529,0.677,0.615,0.459,0.638,0.723];
%KMeansM = [0.178,0.201,0.173,0.177,0.232,0.174,0.178,0.224,0.175,0.179];
% 0.9
KMeans =   [1.458,0.979,1.195,0.569,1.088,0.982,0.935,1.09,0.773,0.763];
KMeansPlus =  [0.749,0.452,0.679,0.608,0.756,0.821,0.778,0.732,0.464,0.643];
KMeansM = [0.204,0.190,0.421,0.202,0.404,0.416,0.448,0.437,0.443,0.400];
mean(KMeans),std(KMeans)
mean(KMeansPlus),std(KMeansPlus)
mean(KMeansM),std(KMeansM)
plot(x,KMeans,'-+',x,KMeansPlus,'-x',x,KMeansM,'-*')
legend('KMeans','KMeans++','KMeansM');
xlabel("Number of Iterations");
ylabel("Time(s)");
grid on