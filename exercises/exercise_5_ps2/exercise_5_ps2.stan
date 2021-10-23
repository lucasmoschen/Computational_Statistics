functions {
   real sum_two_binomials_lpmf(int[] z, real theta1, real theta2, int[] m, int[] n, int T){
       vector[T] z_density;
       vector[max(z)+1] z_i_density;
       int z_i;
       int m_i;
       int n_i;
       int c;
       for(i in 1:T){
            z_i = z[i];
            m_i = m[i];
            n_i = n[i];
            c = 0;
            for(j in max(0,z_i-n_i):min(m_i, z_i)){
                c += 1;
                z_i_density[c] = lchoose(m_i, j) + lchoose(n_i, z_i-j) 
                                 + j*log(theta1) + (m_i-j)*log1m(theta1) 
                                 + (z_i-j)*log(theta2) + (n_i-z_i+j)*log1m(theta2);
            }
            z_density[i] = log_sum_exp(z_i_density[1:c]);  
       }
       return sum(z_density);
   }
}
data {
   int<lower = 0> T;
   int<lower = 0> Z_max;
   int<lower = 0> m[T]; 
   int<lower = 0> n[T]; 
   int<lower = 0, upper = Z_max> z[T];
}
parameters {
   real<lower = 0, upper = 1> theta1; 
   real<lower = 0, upper = 1> theta2;
}   
model {
    theta1 ~ uniform(0,1);
    theta2 ~ uniform(0,1);
    z ~ sum_two_binomials(theta1, theta2, m, n, T);   
}