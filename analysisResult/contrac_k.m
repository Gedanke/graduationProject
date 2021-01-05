clear on;
close all;
x=1:10;
% 0.7
%KMeans =   [2.73,1.88,1.67,2.97,1.95,1.81,2.54,2.62,2.30,2.43];
%KMeansPlus =  [2.41,0.80,1.73,1.57,1.32,2.04,1.00,1.23,1.29,1.87];
%KMeansM = [0.92,0.96,0.95,0.89,0.971,0.975,0.931,0.936,0.95,0.96];
% 0.8
%KMeans =   [2.32,2.59,2.23,2.50,2.17,1.48,1.89,1.40,1.55,1.59]
%KMeansPlus =  [1.51,1.40,1.62,1.53,1.38,0.981,0.985,1.64,1.29,1.23]
%KMeansM = [0.78,0.82,0.77,0.80,0.82,0.77,0.82,0.85,0.78,0.77]
% 0.9
KMeans =   [3.44,1.88,3.22,2.61,1.93,2.35,2.02,1.84,2.38,2.81];
KMeansPlus =  [1.57,1.15,1.66,1.33,1.59,2.29,1.22,2.13,1.35,2.12];
KMeansM = [1.00,1.07,0.98,1.27,1.01,1.11,1.02,1.10,1.03,1.17];
mean(KMeans),std(KMeans)
mean(KMeansPlus),std(KMeansPlus)
mean(KMeansM),std(KMeansM)
plot(x,KMeans,'-+',x,KMeansPlus,'-x',x,KMeansM,'-*')
legend('KMeans','KMeans++','KMeansM');
xlabel("Number of Iterations");
ylabel("Time(s)");
grid on