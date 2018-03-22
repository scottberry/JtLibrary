
function [occupancy_image] = coordinates_to_occupancy_image(coords, im_size)

    % discretize coordinates
    coords_int = round(coords);

    % count occurences at X Y positions
    [a, ~, b] = unique(coords_int(:,[1 2]), 'rows', 'stable');
    tally = accumarray(b, 1);
    xy_count = [a tally];

    % convert occurences to 2D image
    occupancy_image = zeros(im_size);
    linx_occurence = arrayfun(@(x,y) sub2ind(im_size,x,y), xy_count(:,2), xy_count(:,1));
    occupancy_image(linx_occurence) = xy_count(:,3);
end