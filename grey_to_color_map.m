% variables
name = '/home/jzelek/Documents/repos/cancer_similarity/my.png';
save_path = '/home/jzelek/Documents/repos/cancer_similarity/';

% read image, change to color, show and save
img = imread(name);
rgbImage = ind2rgb(img,jet(256));
imshow(rgbImage);
imwrite(rgbImage, strcat(save_path, 'my_colorMapped.png'));