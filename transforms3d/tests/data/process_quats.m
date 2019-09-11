% Load, process, save quaternions.
% Needs the Aerospace toolbox (sorry about that).
% Write needed quats.mat file from Python with the following from this
% directory:
%
% >>> import scipy.io as sio
% >>> from transforms3d.tests.test_quaternions import quats, unit_quats
% >>> sio.savemat('quats.mat', dict(quats=list(quats), unit_quats=list(unit_quats)))
load quats;
quat_e = quatexp(quats);
quat_p = {};
powers = ones(length(unit_quats), 1);
for p = 1:0.5:4
    quat_p{end + 1} = quatpower(unit_quats, powers * p);
end
save processed_quats -v7 quats unit_quats quat_e quat_p;
