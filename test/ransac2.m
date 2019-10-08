close all
clear all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Ground truth %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Set true trajectory 
Nsamples=100;
dt = .01;
t=0:dt:dt*Nsamples;
Vtrue = 2;

% Xtrue is a vector of true positions of the train 
Xinitial = 5;
Xtrue = Xinitial + Vtrue * t.*1.*sin(2.*t);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Motion equations %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Previous state (initial guess): Our guess is that the train starts at 0 with velocity
% that equals to 50% of the real velocity
Xk_prev = [0; 
    .5*Vtrue];

% Current state estimate
Xk=[];

% Motion equation: Xk = Phi*Xk_prev + Noise, that is Xk(n) = Xk(n-1) + Vk(n-1) * dt
% Of course, V is not measured, but it is estimated
% Phi represents the dynamics of the system: it is the motion equation
Phi = [1 dt;
       0  1];

% The error matrix (or the confidence matrix): P states whether we should 
% give more weight to the new measurement or to the model estimate 
sigma_model = 1;
% P = sigma^2*G*G';
P = [sigma_model^2             0;
                 0 sigma_model^2];

% Q is the process noise covariance. It represents the amount of
% uncertainty in the model. In our case, we arbitrarily assume that the model is perfect (no
% acceleration allowed for the train, or in other words - any acceleration is considered to be a noise)
Q = [0 0;
     0 0];

% M is the measurement matrix. 
% We measure X, so M(1) = 1
% We do not measure V, so M(2)= 0
M = [1 0];

% R is the measurement noise covariance. Generally R and sigma_meas can
% vary between samples. 
sigma_meas = 1; % 1 m/sec
R = sigma_meas^2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Kalman iteration %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Buffers for later display
Xk_buffer = zeros(2,Nsamples+1);
Xk_buffer(:,1) = Xk_prev;
Z_buffer = zeros(1,Nsamples+1);

for k=1:Nsamples
    
    % Z is the measurement vector. In our
    % case, Z = TrueData + RandomGaussianNoise
    Z = Xtrue(k+1)+sigma_meas*randn;
    Z_buffer(k+1) = Z;
    
    % Kalman iteration
    P1 = Phi*P*Phi' + Q;
    S = M*P1*M' + R;
    
    % K is Kalman gain. If K is large, more weight goes to the measurement.
    % If K is low, more weight goes to the model prediction.
    K = P1*M'*inv(S);
    P = P1 - K*M*P1;
    
    Xk = Phi*Xk_prev + K*(Z-M*Phi*Xk_prev);
    Xk_buffer(:,k+1) = Xk;
    
    % For the next iteration
    Xk_prev = Xk; 
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Plot resulting graphs %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Position analysis %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;
plot(t,Xtrue,'g');
hold on;
plot(t,Z_buffer,'c');
plot(t,Xk_buffer(1,:),'m');
title('Position estimation results');
xlabel('Time (s)');
ylabel('Position (m)');
legend('True position','Measurements','Kalman estimated displacement');