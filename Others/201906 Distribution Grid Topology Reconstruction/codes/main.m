clc;

v = zeros(1344, 26);

for i = 1: 1344
    for j = 1: 26
        v(i, j) = abs( v_vec(i,j) )+ exp(theta_vec(i,j)*sqrt(-1)); %* exp(sqrt(-1) * theta_vec(i,j));
    end
end


MI = zeros(26, 26);

for i = 2: 26
    for j = i: 26
        fprintf("---\n\n", j);
        MI(i,j) = MI_vol(v(:, i), v(:, j), 1344);
        fprintf("%d,%d,%d\n", i, j, MI(i,j));
        MI(j,i) = MI_vol(v(:, j), v(:, i), 1344);
        fprintf("%d,%d,%d\n", i, j, MI(j,i));
    end
end