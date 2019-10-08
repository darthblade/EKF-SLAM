clear;
clc;

% Generate time vector
dt = 0.25;
t_stop = 100;
t = 0:dt:t_stop;

% Standard deviations of noise to add to each parameter
sd_v = 0;
sd_omega = 0;
sd_r = 0;
sd_th = 0;%2*pi/180;

% Generate input vectors
v =  0.001* t;%ones(size(t)); %0.02 * (t)
omega = 0.5*cos(0.19*t+1);%0*ones(size(t)); %

% Generate position vectors
x = zeros(size(t));
y = zeros(size(t));
phi = zeros(size(t));

% Set starting position
x(1) = 0;
y(1) = 0;
phi(1) = 0;

% Simulate movement
for idx = 2:length(t)
    x(idx) = x(idx-1) + v(idx-1)*dt*cos(phi(idx-1));
    y(idx) = y(idx-1) + v(idx-1)*dt*sin(phi(idx-1));
    phi(idx) = phi(idx-1) + omega(idx-1)*dt;
end

% Specify landmarks
lm = [1 1 1; 2 -2 -0];

% Calculate observations
r = zeros(size(lm, 2), length(t));
theta = zeros(size(lm, 2), length(t));

for idx = 1:length(t)
    x_r = x(idx); y_r = y(idx); phi_r = phi(idx);
    dx = lm(1,:) - x_r;
    dy = lm(2,:) - y_r;
    r(:, idx) = sqrt(dx.^2 + dy.^2)' + sd_r .* randn(size(lm, 2), 1);
    theta(:, idx) = atan2(dy, dx)' - phi_r + sd_th .* randn(size(lm, 2), 1);
end

odometryData=[x' y' phi' v' omega' t'];
rData=[r'];
thetaData=[theta'];
csvwrite('test\rObs.dat',rData)
csvwrite('test\pos.dat',odometryData)
csvwrite('test\thetaObs.dat',thetaData)

comet(x, y);