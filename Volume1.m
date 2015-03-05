%% Read in Files
file_path = 'Pineapple_Images/';
filenames = dir(file_path);
filenames = filenames(4:13);

i = 1;
% read in position one volume 1
fid1 = fopen(strcat('Pineapple_Images/',filenames(i).name));
vol1 = zeros(32,32,32);
for k=1:32
    vol1(:,:,k) = fread(fid1,[32 32],'float');
end
fclose(fid1);

% read in position two volume 1
fid2 = fopen(strcat('Pineapple_Images/',filenames(i+5).name));
vol2 = zeros(32,32,32);
for k=1:32
    vol2(:,:,k) = fread(fid2,[32 32],'float');
end
fclose(fid2);

ang1 = -5.5;
ang2 = -4.5;
theta = ang1:.1:ang2;
cf_ssd = zeros(1,length(theta));

for i = 1:length(theta)
    disp(['Rotation ' num2str(i) ' (' num2str(theta(i)) '°) out of ' num2str(length(theta)) ' ...'])
    J = imrotate(vol2, theta(i), 'bicubic', 'crop');
    cf_ssd(i) = sum((J(:) - vol1(:)).^2);
end

[~, i] = min(cf_ssd);
angMin = theta(i);
plot(theta, cf_ssd)
xlabel('Angle (°)')
ylabel('SSD Cost function')