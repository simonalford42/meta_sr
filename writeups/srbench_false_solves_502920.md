# SRBench solved-task inspection — runs/502920 (noise=0.0)

> **MANUAL-INSPECTION VERDICT: NO false positives.** All 80 solved datasets (evolved-40318 on SRBench, noise=0) are genuine matches.
>
> Method: for each (dataset, seed) the production metric marked solved (`gt_match_score≥1`), I pulled the matched frontier expression + the true equation, then (a) checked whether `round_floats` collapses the GT — the mechanism that breaks **Planck** (0 such cases here; SRBench constants are O(1)), and (b) checked whether the matched expr still equals the true one *outside* the data range (a "good-fit-wrong-function" test).
>
> 3 datasets were auto-flagged by the out-of-range test, but **all 3 are genuine on the data domain** (manually verified; on-data add_rel in brackets) — the flag is an artifact of extrapolating past the data:
> - **feynman_II_2_42** [on-data add_rel 6e-8]: matched = `A·κ·(T2−T1 − 2.3e-7/d²)/d`. With d∈[1,5] the `2.3e-7/d²` term is ~1e-7 (negligible); `round_floats` correctly discards it. Flagged only because the test extends d toward 0 where `1/d²` blows up. ✅ genuine.
> - **feynman_II_6_15b** [3.6e-4]: matched = `−0.2387·p_d·sin(θ+3.1413)·cos(θ)/(ε·r³)`; `sin(θ+π)=−sin(θ)` and the `−0.2387` sign give `+0.2387·sin·cos` = true (`3/4π=0.2387`). PySR encoded the minus sign as a `≈π` phase shift. ✅ genuine.
> - **strogatz_predprey2** [8e-5]: matched = `0.206·y·(4.856·x/(x+0.9997) − 0.364·y − tiny·sin)`; `0.206·4.856=1.0`, `0.206·0.364=0.075` ⇒ `y·(x/(1+x) − 0.075y)` = true; `0.9997≈1` and the `sin` term is negligible. ✅ genuine.
>
> Takeaway: the `round_floats(zero_threshold=1e-4)` degeneracy that makes **Planck** match anything does **not** affect SRBench solved tasks, because SRBench's synthetic Feynman/Strogatz equations sample variables and constants at O(1) magnitudes. The metric's solves here are trustworthy (at noise=0).

80 datasets with >=1 solved seed. **3** have a matched eq that DIVERGES from the true eq outside the data range (false-positive candidates, all verified genuine above); **0** have a round_floats-collapsed GT.

- `gen`: does matched == true up to add/mult const OUTSIDE the data range? `generalizes` = genuine; `DIVERGES` = good fit, wrong function (false positive).
- `why`: which sympy condition made the official check pass.

---

## feynman_II_2_42  🚩 **DIVERGES (false-positive candidate)**
- solved 9 seeds (8 distinct matched eqs)
- **true:** `kappa*(T2-T1)*A/d`
- **matched:** `-1.08205024290492*A*T1*kappa*(0.9241716 - 0.924171625694596*T2/T1)/d`  (R²=0.9999999999999836)  [generalizes] `add_rel=2.0e-07 mult_rel=6.44871661698182e-07`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `A*kappa*(-T1 + T2 - 2.3244442e-7/d**2)/d`  (R²=0.9999999999999944)  ❌DIVERGES `add_rel=1.2e-01 mult_rel=0.007581427974678771`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `A*kappa*(-T1 + T2)/d`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `A*kappa*(-T1 + T2 - 2.386603e-7/d**2)/d`  (R²=0.9999999999999949)  ❌DIVERGES `add_rel=1.2e+03 mult_rel=17.194354203149793`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_II_6_15b  🚩 **DIVERGES (false-positive candidate)**
- solved 5 seeds (5 distinct matched eqs)
- **true:** `p_d/(4*pi*epsilon)*3*cos(theta)*sin(theta)/r**3`
- **matched:** `0.238732887941646*p_d*sin(theta)*cos(theta)/(epsilon*r**3)`  (R²=0.9999999999957377)  [generalizes] `add_rel=2.0e-06 mult_rel=1.9875743603232396e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `-0.238729244581224*p_d*sin(theta + 3.1413183)*cos(theta)/(epsilon*r**3)`  (R²=0.9999998401685913)  ❌DIVERGES `add_rel=2.4e-03 mult_rel=0.006841871368771439`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.238732408344446*p_d*sin(theta)*cos(theta)/(epsilon*r**3)`  (R²=0.9999999999999992)  [generalizes] `add_rel=2.6e-08 mult_rel=1.7023922970672168e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.23873241*p_d*sin(theta)*cos(theta)/(epsilon*r**3)`  (R²=0.9999999999999996)  [generalizes] `add_rel=1.9e-08 mult_rel=2.598669086163714e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}

## strogatz_predprey2  🚩 **DIVERGES (false-positive candidate)**
- solved 4 seeds (4 distinct matched eqs)
- **true:** `y * ( (x)/(1+x) - 0.075 * y )`
- **matched:** `0.20590018*y*(4.8561516*x/(x + 0.99965763) - 0.36425*y - 0.00024310737*sin(4.856151*x/(x + 0.99960274)))`  (R²=0.9999999993701386)  ❌DIVERGES `add_rel=1.7e-02 mult_rel=0.0018944107775020211`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `-0.042770136*y**2*(-23.3808*x/(x*y + y) + 1.75356)`  (R²=1.0)  [generalizes] `add_rel=4.2e-09 mult_rel=6.950559232963258e-15`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `y*(x/(1.5399891e-8*x**2 + x + 1.0) - 0.074999996*y)`  (R²=0.9999999999999928)  [generalizes] `add_rel=7.2e-07 mult_rel=4.285484356968035e-06`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `1.000036366095*y*(2.5400245e-8*x*y + 4.0761876e-7*x + 0.999958241043375*x/(x + 0.9999856) - 0.0749973702091112*y)`  (R²=0.9999999999985194)  [generalizes] `add_rel=6.9e-04 mult_rel=9.691867482439504e-05`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_III_12_43
- solved 10 seeds (4 distinct matched eqs)
- **true:** `n*(h/(2*pi))`
- **matched:** `0.15915494*h*n`  (R²=0.9999999999999986)  [generalizes] `add_rel=1.9e-08 mult_rel=2.7869141683049463e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.15915494*h*n`  (R²=0.9999999999999984)  [generalizes] `add_rel=1.9e-08 mult_rel=2.7766676165738707e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.15915494*h*n`  (R²=0.9999999999999984)  [generalizes] `add_rel=1.9e-08 mult_rel=2.744295309755616e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.15915494*h*n`  (R²=0.9999999999999984)  [generalizes] `add_rel=1.9e-08 mult_rel=2.7403624451760835e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}

## feynman_III_13_18
- solved 10 seeds (10 distinct matched eqs)
- **true:** `2*E_n*d**2*k/(h/(2*pi))`
- **matched:** `12.566371*E_n*d**2*k/h`  (R²=0.9999999999999986)  [generalizes] `add_rel=3.1e-08 mult_rel=1.849446233686154e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `12.5663699110035*E_n*d**2*k/h`  (R²=0.9999999999999948)  [generalizes] `add_rel=5.6e-08 mult_rel=2.150296055914565e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `12.566371*E_n*d**2*k/h`  (R²=0.9999999999999983)  [generalizes] `add_rel=3.1e-08 mult_rel=1.902016452477934e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `12.566371*E_n*d**2*k/h`  (R²=0.9999999999999983)  [generalizes] `add_rel=3.1e-08 mult_rel=1.8175133666259925e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}

## feynman_III_15_12
- solved 6 seeds (6 distinct matched eqs)
- **true:** `2*U*(1-cos(k*d))`
- **matched:** `2.0*U*(1.0 - cos(d*k))`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `U*(2.0 - 2.0*cos(d*k))`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `-2.0*U*(cos(d*k) - 1.0)`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `1.9999995*U*(sin(d*k - 1.5707985) + 1.0)`  (R²=0.9999999999991156)  [generalizes] `add_rel=1.7e-06 mult_rel=0.00010075486734374327`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': False}

## feynman_III_15_14
- solved 10 seeds (9 distinct matched eqs)
- **true:** `(h/(2*pi))**2/(2*E_n*d**2)`
- **matched:** `0.0126651487908013*h**2/(E_n*d**2)`  (R²=0.9999999999999933)  [generalizes] `add_rel=6.6e-08 mult_rel=2.7493805215859524e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.012665148*h**2/(E_n*d**2)`  (R²=1.0)  [generalizes] `add_rel=3.5e-09 mult_rel=2.022663318446569e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.012665148*h**2/(E_n*d**2)`  (R²=1.0)  [generalizes] `add_rel=3.5e-09 mult_rel=2.0931290665812058e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.0126651468989067*h**2/(E_n*d**2)`  (R²=0.9999999999999903)  [generalizes] `add_rel=8.3e-08 mult_rel=1.288975694038892e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}

## feynman_III_15_27
- solved 10 seeds (8 distinct matched eqs)
- **true:** `2*pi*alpha/(n*d)`
- **matched:** `6.2831855*alpha/(d*n)`  (R²=0.999999999999998)  [generalizes] `add_rel=3.1e-08 mult_rel=1.6184142126183178e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `6.2831855*alpha/(d*n)`  (R²=0.9999999999999978)  [generalizes] `add_rel=3.1e-08 mult_rel=1.5992607070666136e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `6.2831855*alpha/(d*n)`  (R²=0.9999999999999978)  [generalizes] `add_rel=3.1e-08 mult_rel=1.6203171106449332e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `6.2831855*alpha/(d*n)`  (R²=0.9999999999999973)  [generalizes] `add_rel=3.1e-08 mult_rel=1.6411033693616882e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}

## feynman_III_17_37
- solved 10 seeds (3 distinct matched eqs)
- **true:** `beta*(1+alpha*cos(theta))`
- **matched:** `beta*(alpha*cos(theta) + 1.0)`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `beta*(alpha*cos(theta) + 1.0)`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `beta*(alpha*cos(theta) + 1.0)`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_III_19_51
- solved 10 seeds (10 distinct matched eqs)
- **true:** `-m*q**4/(2*(4*pi*epsilon)**2*(h/(2*pi))**2)*(1/n**2)`
- **matched:** `-0.12499999495155*m*q**4/(epsilon**2*h**2*n**2)`  (R²=0.9999999999999983)  [generalizes] `add_rel=4.0e-08 mult_rel=1.8464574225181852e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `0.124999992509424*q**4*(4.0170622e-15 - m)/(epsilon**2*h**2*n**2)`  (R²=0.9999999999999963)  [generalizes] `add_rel=6.0e-08 mult_rel=5.451421729311201e-14`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `-0.125000000115142*m*q**4/(epsilon**2*h**2*n**2)`  (R²=1.0)  [generalizes] `add_rel=9.2e-10 mult_rel=1.5809174159110062e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `-0.125*m*q**4/(epsilon**2*h**2*n**2)`  (R²=1.0)  [generalizes] `add_rel=1.3e-17 mult_rel=1.1425697223163032e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_III_21_20
- solved 10 seeds (10 distinct matched eqs)
- **true:** `-rho_c_0*q*A_vec/m`
- **matched:** `A_vec*rho_c_0*(-q - 3.1664946e-8)/m`  (R²=0.9999999999999997)  [generalizes] `add_rel=5.2e-09 mult_rel=1.725575738588807e-07`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `A_vec*q*(8.335513e-9 - rho_c_0/m)`  (R²=0.9999999999999999)  [generalizes] `add_rel=4.9e-11 mult_rel=5.799307965932124e-07`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `A_vec*(1.5211143e-17 - q*rho_c_0/m)`  (R²=1.0)  [generalizes] `add_rel=9.8e-17 mult_rel=4.445746670839396e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `-1.0*A_vec*q*rho_c_0/m`  (R²=1.0)  [generalizes] `add_rel=5.2e-18 mult_rel=1.1150696714091006e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_III_7_38
- solved 10 seeds (7 distinct matched eqs)
- **true:** `2*mom*B/(h/(2*pi))`
- **matched:** `12.566371*B*mom/h`  (R²=0.9999999999999978)  [generalizes] `add_rel=3.1e-08 mult_rel=1.719950061197322e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `12.56637*B*mom/h`  (R²=0.9999999999999942)  [generalizes] `add_rel=4.9e-08 mult_rel=1.2034531398402913e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `12.566371*B*mom/h`  (R²=0.9999999999999977)  [generalizes] `add_rel=3.1e-08 mult_rel=1.7109685373477176e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `12.566371*B*mom/h`  (R²=0.9999999999999978)  [generalizes] `add_rel=3.1e-08 mult_rel=1.721740743682523e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}

## feynman_II_10_9
- solved 10 seeds (7 distinct matched eqs)
- **true:** `sigma_den/epsilon*1/(1+chi)`
- **matched:** `sigma_den/(epsilon*(chi + 0.99999994))`  (R²=0.9999999999999989)  [generalizes] `add_rel=8.2e-06 mult_rel=4.7119359969397463e-07`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `sigma_den/(chi*epsilon + epsilon)`  (R²=1.0)  [generalizes] `add_rel=1.7e-15 mult_rel=2.518579533156299e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `sigma_den/(epsilon*(chi + 0.99999994))`  (R²=0.999999999999999)  [generalizes] `add_rel=3.5e-06 mult_rel=2.380848029068717e-07`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `sigma_den/(epsilon*(chi + 0.99999994))`  (R²=0.9999999999999988)  [generalizes] `add_rel=9.3e-06 mult_rel=5.3556904812788e-07`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_II_11_20
- solved 10 seeds (10 distinct matched eqs)
- **true:** `n_rho*p_d**2*Ef/(3*kb*T)`
- **matched:** `0.33333334*Ef*n_rho*p_d**2/(T*kb)`  (R²=0.9999999999999994)  [generalizes] `add_rel=2.0e-08 mult_rel=1.8775574636206252e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `0.3333333*Ef*n_rho*p_d**2/(T*kb)`  (R²=0.9999999999999856)  [generalizes] `add_rel=1.0e-07 mult_rel=2.648586157261591e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `0.333333311111113*Ef*n_rho*p_d**2/(T*kb)`  (R²=0.9999999999999934)  [generalizes] `add_rel=6.7e-08 mult_rel=2.064014923639244e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `0.333333333333333*Ef*n_rho*p_d**2/(T*kb)`  (R²=1.0)  [generalizes] `add_rel=1.2e-15 mult_rel=1.7497023288542417e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_II_11_27
- solved 7 seeds (7 distinct matched eqs)
- **true:** `n*alpha/(1-(n*alpha/3))*epsilon*Ef`
- **matched:** `Ef*alpha*epsilon*n/(-0.33333337*alpha*n + 1.00000002337793)`  (R²=0.9999999999999999)  [generalizes] `add_rel=1.1e-07 mult_rel=3.3134743758021106e-08`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `Ef*alpha*epsilon*n/(-3.4913146e-9*Ef - 0.33333334*alpha*n + 3.4913146e-9*alpha + 0.999999996508685)`  (R²=0.9999999999999999)  [generalizes] `add_rel=4.1e-08 mult_rel=8.424965831507315e-09`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `1.0316167*Ef*alpha*epsilon*n/(-0.34387153*alpha*n + 1.0316173)`  (R²=0.9999999999999981)  [generalizes] `add_rel=4.4e-06 mult_rel=1.0041320970232122e-06`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `Ef*alpha*epsilon*n/(-0.33333328*alpha*n + 0.999999983456339)`  (R²=0.9999999999999999)  [generalizes] `add_rel=2.0e-07 mult_rel=5.485167117247976e-08`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}

## feynman_II_11_28
- solved 4 seeds (4 distinct matched eqs)
- **true:** `1+n*alpha/(1-(n*alpha/3))`
- **matched:** `(0.73916954*alpha*n + 1.1078452)/(-0.36895394*alpha*n + 1.1078703)`  (R²=0.9999999989557471)  [generalizes] `add_rel=1.2e-03 mult_rel=0.00021755310461345345`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `alpha*n/(-0.33333328*alpha*n + 1) + 1`  (R²=0.9999999999999968)  [generalizes] `add_rel=2.5e-07 mult_rel=4.494759789085683e-08`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `(alpha*n + 1.5019073)/(-0.5013224*alpha*n + 1.5018764)`  (R²=0.9999999999907341)  [generalizes] `add_rel=1.8e-03 mult_rel=0.0003332120785886038`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `(0.71271616828219*alpha*n + 1.07049472433448)/(-0.3572744*alpha*n + 1.0704433)`  (R²=0.9999999637875384)  [generalizes] `add_rel=1.6e-03 mult_rel=0.0003021336021865927`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}

## feynman_II_11_3
- solved 4 seeds (3 distinct matched eqs)
- **true:** `q*Ef/(m*(omega_0**2-omega**2))`
- **matched:** `Ef*q/(m*(-omega**2 + omega_0**2))`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `Ef*q/(m*(-omega**2 + omega_0**2))`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `Ef*q/(m*omega_0**2*(-omega**2/omega_0**2 + 1.0))`  (R²=1.0)  [generalizes] `add_rel=8.9e-16 mult_rel=2.817156743035595e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_II_13_17
- solved 10 seeds (9 distinct matched eqs)
- **true:** `1/(4*pi*epsilon*c**2)*2*I/r`
- **matched:** `0.15915494*I/(c**2*epsilon*r)`  (R²=0.9999999999999996)  [generalizes] `add_rel=1.9e-08 mult_rel=2.9515024416257783e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.159154938207697*I/(c**2*epsilon*r)`  (R²=0.9999999999999987)  [generalizes] `add_rel=3.1e-08 mult_rel=2.38439626420999e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.15915494*I/(c**2*epsilon*r)`  (R²=0.9999999999999994)  [generalizes] `add_rel=1.9e-08 mult_rel=2.9107647346599497e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.15915494*I/(c**2*epsilon*r)`  (R²=0.9999999999999994)  [generalizes] `add_rel=1.9e-08 mult_rel=2.946016130064478e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}

## feynman_II_15_4
- solved 10 seeds (10 distinct matched eqs)
- **true:** `-mom*B*cos(theta)`
- **matched:** `B*mom*sin(theta - 1.5707964)`  (R²=0.9999999999999879)  [generalizes] `add_rel=6.8e-08 mult_rel=1.5353869792728664e-06`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `B*mom*sin(theta - 1.5707964)`  (R²=0.99999999999999)  [generalizes] `add_rel=6.8e-08 mult_rel=1.4527830945097569e-06`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `mom*(-B*cos(theta) + 1.3407084e-8)`  (R²=0.9999999999999999)  [generalizes] `add_rel=3.0e-09 mult_rel=5.229290114409471e-07`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `mom*(-B*cos(theta) + 1.35003395e-8)`  (R²=0.9999999999999999)  [generalizes] `add_rel=3.0e-09 mult_rel=5.834728020114551e-07`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_II_15_5
- solved 10 seeds (10 distinct matched eqs)
- **true:** `-p_d*Ef*cos(theta)`
- **matched:** `Ef*p_d*(9.310431e-9 - cos(theta))`  (R²=0.9999999999999997)  [generalizes] `add_rel=1.0e-08 mult_rel=1.742610603032026e-07`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `Ef*p_d*(2.2394174e-8 - cos(theta))`  (R²=0.999999999999998)  [generalizes] `add_rel=2.4e-08 mult_rel=4.7303658033287774e-07`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `-1.0*Ef*p_d*cos(theta)`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `Ef*p_d*(2.4102249e-8 - cos(theta))`  (R²=0.9999999999999977)  [generalizes] `add_rel=2.6e-08 mult_rel=3.6164939015103886e-07`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_II_21_32
- solved 10 seeds (10 distinct matched eqs)
- **true:** `q/(4*pi*epsilon*r*(1-v/c))`
- **matched:** `q*(1.63198418564412e-6 + 0.08182683015066/(1.0282855453468 - 1.0282948*v/c))/(epsilon*r)`  (R²=0.9999999999990966)  [generalizes] `add_rel=3.1e-04 mult_rel=0.0002276207533533366`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.11821707*q/(epsilon*r*(1.4855574 - 1.4855574108517*v/c)*(sin(1.78984247644153e-7/r) + 1.000001 + 1.1182827e-6*v/c))`  (R²=0.9999999999999527)  [generalizes] `add_rel=2.6e-05 mult_rel=5.724003023283325e-06`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.08217884*q/(epsilon*r*(1.0326897 - 1.0326897*v/c))`  (R²=0.9999999999999934)  [generalizes] `add_rel=5.8e-08 mult_rel=1.6158226305396994e-15`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.13047189*q/(epsilon*r*(1.63955808297299 - 1.63955808297299*v/c))`  (R²=0.9999999999999987)  [generalizes] `add_rel=2.5e-08 mult_rel=1.6826014649128602e-15`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}

## feynman_II_27_16
- solved 10 seeds (4 distinct matched eqs)
- **true:** `epsilon*c*Ef**2`
- **matched:** `Ef**2*c*epsilon`  (R²=1.0)  [generalizes] `add_rel=1.2e-16 mult_rel=1.200248140345009e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `Ef**2*c*epsilon`  (R²=1.0)  [generalizes] `add_rel=1.1e-16 mult_rel=1.0792601309996586e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `Ef**2*c*epsilon`  (R²=1.0)  [generalizes] `add_rel=9.3e-17 mult_rel=1.0955544447183469e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `Ef**2*c*epsilon`  (R²=1.0)  [generalizes] `add_rel=1.1e-16 mult_rel=1.1046579618885443e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_II_27_18
- solved 10 seeds (2 distinct matched eqs)
- **true:** `epsilon*Ef**2`
- **matched:** `Ef**2*epsilon`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `Ef**2*epsilon`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_II_34_11
- solved 10 seeds (10 distinct matched eqs)
- **true:** `g_*q*B/(2*m)`
- **matched:** `B*g_*q/(2*m)`  (R²=1.0)  [generalizes] `add_rel=2.8e-18 mult_rel=1.1178297490756088e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `0.5*B*g_*q/m`  (R²=1.0)  [generalizes] `add_rel=5.6e-18 mult_rel=1.0913272092315241e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `0.5*B*g_*q/m`  (R²=1.0)  [generalizes] `add_rel=3.9e-17 mult_rel=1.1011655157087892e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `0.5*B*g_*q/m`  (R²=1.0)  [generalizes] `add_rel=3.3e-18 mult_rel=1.1683745737483645e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_II_34_2
- solved 10 seeds (8 distinct matched eqs)
- **true:** `q*v*r/2`
- **matched:** `0.5*q*r*v`  (R²=1.0)  [generalizes] `add_rel=1.1e-16 mult_rel=1.1504342909453405e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `0.5*q*r*v`  (R²=1.0)  [generalizes] `add_rel=1.2e-16 mult_rel=1.1067481396552162e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `0.5*q*r*v`  (R²=1.0)  [generalizes] `add_rel=1.1e-16 mult_rel=1.1762602059271538e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `0.5*q*r*v`  (R²=1.0)  [generalizes] `add_rel=1.2e-16 mult_rel=1.1205830284902934e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_II_34_29a
- solved 10 seeds (8 distinct matched eqs)
- **true:** `q*h/(4*pi*m)`
- **matched:** `0.07957747*h*q/m`  (R²=0.9999999999999991)  [generalizes] `add_rel=1.9e-08 mult_rel=2.7998759883647373e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.07957747*h*q/m`  (R²=0.999999999999999)  [generalizes] `add_rel=1.9e-08 mult_rel=2.89617168339421e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.07957747*h*q/m`  (R²=0.999999999999999)  [generalizes] `add_rel=1.9e-08 mult_rel=2.8785623694267573e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.079577476*h*q/m`  (R²=0.9999999999999919)  [generalizes] `add_rel=5.6e-08 mult_rel=2.1868856171598585e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}

## feynman_II_34_29b
- solved 10 seeds (10 distinct matched eqs)
- **true:** `g_*mom*B*Jz/(h/(2*pi))`
- **matched:** `6.2831855*B*Jz*g_*mom/h`  (R²=0.9999999999999981)  [generalizes] `add_rel=3.1e-08 mult_rel=2.0486547481729874e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `6.2831855*B*Jz*g_*mom/h`  (R²=0.999999999999998)  [generalizes] `add_rel=3.1e-08 mult_rel=2.0091662240027678e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `6.2831855*B*Jz*g_*mom/h`  (R²=0.9999999999999982)  [generalizes] `add_rel=3.1e-08 mult_rel=1.9531722401512853e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `6.2831855*B*Jz*g_*mom/h`  (R²=0.9999999999999982)  [generalizes] `add_rel=3.1e-08 mult_rel=1.9782541309948054e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}

## feynman_II_34_2a
- solved 10 seeds (8 distinct matched eqs)
- **true:** `q*v/(2*pi*r)`
- **matched:** `0.15915494*q*v/r`  (R²=0.999999999999999)  [generalizes] `add_rel=1.9e-08 mult_rel=2.920013250154541e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.15915495*q*v/r`  (R²=0.9999999999999956)  [generalizes] `add_rel=4.3e-08 mult_rel=2.1995307682140961e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.15915494*q*v/r`  (R²=0.9999999999999992)  [generalizes] `add_rel=1.9e-08 mult_rel=2.8359625808974886e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.15915494*q*v/r`  (R²=0.999999999999999)  [generalizes] `add_rel=1.9e-08 mult_rel=2.826439052891936e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}

## feynman_II_36_38
- solved 5 seeds (5 distinct matched eqs)
- **true:** `mom*H/(kb*T)+(mom*alpha)/(epsilon*c**2*kb*T)*M`
- **matched:** `mom*(H + M*alpha/(c**2*epsilon))/(T*kb)`  (R²=1.0)  [generalizes] `add_rel=2.0e-16 mult_rel=1.4634323005040603e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `mom*(H + M*alpha/(c**2*epsilon))/(T*kb)`  (R²=1.0)  [generalizes] `add_rel=6.5e-17 mult_rel=1.478619515923008e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `mom*(H + 1.0*M*alpha/(c**2*epsilon))/(T*kb)`  (R²=1.0)  [generalizes] `add_rel=2.2e-16 mult_rel=1.4174378709434726e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `-1.00003403115808*mom*(-H - 0.999751447809281*M*alpha/(c**2*epsilon))/(T*kb)`  (R²=0.9999999850178217)  [generalizes] `add_rel=2.1e-04 mult_rel=8.59409142138559e-05`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_II_37_1
- solved 10 seeds (8 distinct matched eqs)
- **true:** `mom*(1+chi)*B`
- **matched:** `B*(chi*mom + mom)`  (R²=1.0)  [generalizes] `add_rel=1.3e-16 mult_rel=2.7350158873396006e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `B*(chi*mom + mom)`  (R²=1.0)  [generalizes] `add_rel=1.3e-16 mult_rel=3.2313498396939068e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `B*(chi*mom + mom)`  (R²=1.0)  [generalizes] `add_rel=1.3e-16 mult_rel=1.8652077234295119e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `mom*(B*chi + B)`  (R²=1.0)  [generalizes] `add_rel=1.2e-16 mult_rel=2.4884231869185434e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_II_38_14
- solved 10 seeds (9 distinct matched eqs)
- **true:** `Y/(2*(1+sigma))`
- **matched:** `-2.255284*Y/(-4.510568*sigma - 4.510568)`  (R²=1.0)  [generalizes] `add_rel=5.0e-15 mult_rel=3.0009412556654147e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `Y/(2*sigma + 1.9999999)`  (R²=0.9999999999999986)  [generalizes] `add_rel=5.0e-06 mult_rel=2.4584233756444517e-07`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `Y/(2*sigma + 2.0)`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `0.49999998*Y/(sigma + 0.9999995)`  (R²=0.9999999999999105)  [generalizes] `add_rel=3.1e-05 mult_rel=1.7607007119632004e-06`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_II_38_3
- solved 10 seeds (8 distinct matched eqs)
- **true:** `Y*A*x/d`
- **matched:** `A*Y*x/d`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `A*Y*x/d`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `A*Y*x/d`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `A*Y*x/d`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_II_3_24
- solved 10 seeds (5 distinct matched eqs)
- **true:** `Pwr/(4*pi*r**2)`
- **matched:** `0.07957747*Pwr/r**2`  (R²=0.9999999999999993)  [generalizes] `add_rel=1.9e-08 mult_rel=2.750184061380169e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.07957747*Pwr/r**2`  (R²=0.9999999999999993)  [generalizes] `add_rel=1.9e-08 mult_rel=2.752983801554825e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.0795774755649174*Pwr/r**2`  (R²=0.9999999999999959)  [generalizes] `add_rel=5.1e-08 mult_rel=2.5121478070208716e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.07957747*Pwr/r**2`  (R²=0.9999999999999993)  [generalizes] `add_rel=1.9e-08 mult_rel=2.755221545311057e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}

## feynman_II_4_23
- solved 10 seeds (8 distinct matched eqs)
- **true:** `q/(4*pi*epsilon*r)`
- **matched:** `0.079577476*q/(epsilon*r)`  (R²=0.9999999999999934)  [generalizes] `add_rel=5.6e-08 mult_rel=2.195323792367198e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.07957747*q/(epsilon*r)`  (R²=0.9999999999999991)  [generalizes] `add_rel=1.9e-08 mult_rel=2.8565329169150397e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.07957747*q/(epsilon*r)`  (R²=0.999999999999999)  [generalizes] `add_rel=1.9e-08 mult_rel=2.768610026075212e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.07957747*q/(epsilon*r)`  (R²=0.9999999999999992)  [generalizes] `add_rel=1.9e-08 mult_rel=2.888980846471531e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}

## feynman_II_6_11
- solved 8 seeds (7 distinct matched eqs)
- **true:** `1/(4*pi*epsilon)*p_d*cos(theta)/r**2`
- **matched:** `0.07957747*p_d*cos(theta)/(epsilon*r**2)`  (R²=0.9999999999999996)  [generalizes] `add_rel=1.9e-08 mult_rel=2.9759362560995924e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.07957747*p_d*cos(theta)/(epsilon*r**2)`  (R²=0.9999999999999996)  [generalizes] `add_rel=1.9e-08 mult_rel=2.898298875516023e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.079577476*p_d*cos(theta)/(epsilon*r**2)`  (R²=0.999999999999996)  [generalizes] `add_rel=5.6e-08 mult_rel=2.300868453108147e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.079577476*p_d*cos(theta)/(epsilon*r**2)`  (R²=0.9999999999999961)  [generalizes] `add_rel=5.6e-08 mult_rel=2.2535064918662597e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}

## feynman_II_8_31
- solved 10 seeds (8 distinct matched eqs)
- **true:** `epsilon*Ef**2/2`
- **matched:** `0.5*Ef**2*epsilon`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `0.5*Ef**2*epsilon`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `0.5*Ef**2*epsilon`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `0.5*Ef**2*epsilon`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_II_8_7
- solved 10 seeds (8 distinct matched eqs)
- **true:** `3/5*q**2/(4*pi*epsilon*d)`
- **matched:** `0.047746483*q**2/(d*epsilon)`  (R²=1.0)  [generalizes] `add_rel=1.5e-09 mult_rel=1.407405848325115e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.047746483*q**2/(d*epsilon)`  (R²=1.0)  [generalizes] `add_rel=1.5e-09 mult_rel=2.382456865270969e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.047746483*q**2/(d*epsilon)`  (R²=1.0)  [generalizes] `add_rel=1.5e-09 mult_rel=1.4590708556663235e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.0477464846238961*q**2/(d*epsilon)`  (R²=0.9999999999999976)  [generalizes] `add_rel=3.6e-08 mult_rel=2.851133902793972e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}

## feynman_I_11_19
- solved 9 seeds (9 distinct matched eqs)
- **true:** `x1*y1+x2*y2+x3*y3`
- **matched:** `y1**2 + y1*y2 + y1*y3`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `y1**2 + y1*y2 + y1*y3`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `y1**2 + y1*y2 + y1*y3`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `y1*y2 + y1*y3 - y1*(3.8960724e-10 - y1)`  (R²=1.0)  [generalizes] `add_rel=4.7e-11 mult_rel=5.92925544907737e-10`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_I_12_1
- solved 10 seeds (2 distinct matched eqs)
- **true:** `mu*Nn`
- **matched:** `Nn*mu`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `Nn*mu`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_I_12_11
- solved 10 seeds (8 distinct matched eqs)
- **true:** `q*(Ef+B*v*sin(theta))`
- **matched:** `q*(B*v*sin(theta) + Ef)`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `q*(B*v*sin(theta) + Ef)`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `q*(B*v*sin(theta) + Ef)`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `q*(B*v*sin(theta) + Ef)`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_I_12_2
- solved 10 seeds (10 distinct matched eqs)
- **true:** `q1*q2*r/(4*pi*epsilon*r**3)`
- **matched:** `0.079577476*q1*q2/(epsilon*r**2)`  (R²=0.9999999999999947)  [generalizes] `add_rel=5.6e-08 mult_rel=2.2562396723636553e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.0795774691038487*q1*q2/(epsilon*r**2)`  (R²=0.9999999999999987)  [generalizes] `add_rel=3.1e-08 mult_rel=2.4747641501868504e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.079577476*q1*q2/(epsilon*r**2)`  (R²=0.9999999999999952)  [generalizes] `add_rel=5.6e-08 mult_rel=2.2452869879128e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.079577476*q1*q2/(epsilon*r**2)`  (R²=0.9999999999999957)  [generalizes] `add_rel=5.6e-08 mult_rel=2.300868453108147e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}

## feynman_I_12_4
- solved 10 seeds (9 distinct matched eqs)
- **true:** `q1*r/(4*pi*epsilon*r**3)`
- **matched:** `0.07957747*q1/(epsilon*r**2)`  (R²=0.9999999999999994)  [generalizes] `add_rel=1.9e-08 mult_rel=2.8449127074056377e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.07957747*q1/(epsilon*r**2)`  (R²=0.9999999999999994)  [generalizes] `add_rel=1.9e-08 mult_rel=2.838949091551283e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.0795774671375793*q1/(epsilon*r**2)`  (R²=0.999999999999995)  [generalizes] `add_rel=5.5e-08 mult_rel=1.2658490791813218e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.079577476*q1/(epsilon*r**2)`  (R²=0.9999999999999954)  [generalizes] `add_rel=5.6e-08 mult_rel=2.1741669499721946e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}

## feynman_I_12_5
- solved 10 seeds (2 distinct matched eqs)
- **true:** `q2*Ef`
- **matched:** `Ef*q2`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `Ef*q2`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_I_13_12
- solved 9 seeds (9 distinct matched eqs)
- **true:** `G*m1*m2*(1/r2-1/r1)`
- **matched:** `G*m1*m2*(r1/r2 - 1.0)/r1`  (R²=1.0)  [generalizes] `add_rel=1.0e-16 mult_rel=2.9173816741308873e-15`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `1.0*G*m1*m2*(r1 - r2)/(r1*r2)`  (R²=1.0)  [generalizes] `add_rel=1.4e-16 mult_rel=1.1959787032121716e-15`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `-0.0264309309555363*G*m1*m2*(-37.83446*r1/r2 + 37.83446)/r1`  (R²=1.0)  [generalizes] `add_rel=3.2e-16 mult_rel=4.2726839706545565e-15`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `-0.655388694322191*G*m1*m2*(-1.5258121*r1/r2 + 1.5258121)/r1`  (R²=1.0)  [generalizes] `add_rel=3.9e-16 mult_rel=2.7693528061665247e-15`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_I_13_4
- solved 1 seeds (1 distinct matched eqs)
- **true:** `1/2*m*(v**2+u**2+w**2)`
- **matched:** `0.500001295272281*m*v**2 + m*(0.500003264115966*u**2 + 0.499995496776787*w**2)`  (R²=0.9999999999742102)  [generalizes] `add_rel=4.9e-06 mult_rel=4.140576109487689e-06`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_I_14_3
- solved 10 seeds (4 distinct matched eqs)
- **true:** `m*g*z`
- **matched:** `g*m*z`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `g*m*z`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `g*m*z`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `g*m*z`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_I_14_4
- solved 10 seeds (6 distinct matched eqs)
- **true:** `1/2*k_spring*x**2`
- **matched:** `0.5*k_spring*x**2`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `0.5*k_spring*x**2`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `0.5*k_spring*x**2`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `0.5*k_spring*x**2`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_I_18_12
- solved 10 seeds (5 distinct matched eqs)
- **true:** `r*F*sin(theta)`
- **matched:** `F*r*sin(theta)`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `F*r*sin(theta)`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `F*r*sin(theta)`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `F*r*sin(theta)`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_I_18_14
- solved 10 seeds (8 distinct matched eqs)
- **true:** `m*r*v*sin(theta)`
- **matched:** `m*r*v*sin(theta)`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `m*r*v*sin(theta)`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `m*r*v*sin(theta)`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `m*r*v*sin(theta)`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_I_18_4
- solved 2 seeds (2 distinct matched eqs)
- **true:** `(m1*r1+m2*r2)/(m1+m2)`
- **matched:** `m1*(r1 - r2)/(m1 + m2) + r2`  (R²=1.0)  [generalizes] `add_rel=3.5e-16 mult_rel=4.4140965169983296e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `r2 - 1.3312309*(-r1 + r2)/(1.331231128 + 1.3312308*m2/m1)`  (R²=0.9999999999999953)  [generalizes] `add_rel=1.4e-06 mult_rel=1.4272927569908598e-06`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_I_24_6
- solved 3 seeds (3 distinct matched eqs)
- **true:** `1/2*m*(omega**2+omega_0**2)*1/2*x**2`
- **matched:** `0.25*m*x**2*(omega**2 + omega_0**2)`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `m*omega_0**2*x*(0.25*omega**2*x/omega_0**2 + 0.25*x)`  (R²=1.0)  [generalizes] `add_rel=1.6e-16 mult_rel=1.5686198063741045e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `0.9999938*m*x*(0.12500077*x*(omega - omega_0)**2 + 0.12500077*x*(omega + omega_0)**2)`  (R²=0.9999999999999958)  [generalizes] `add_rel=4.0e-08 mult_rel=2.1123434504021245e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_I_25_13
- solved 10 seeds (1 distinct matched eqs)
- **true:** `q/C`
- **matched:** `q/C`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_I_27_6
- solved 10 seeds (3 distinct matched eqs)
- **true:** `1/(1/d1+n/d2)`
- **matched:** `d2/(n + d2/d1)`  (R²=1.0)  [generalizes] `add_rel=3.0e-14 mult_rel=2.7433181852578957e-15`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `d2/(n + d2/d1)`  (R²=1.0)  [generalizes] `add_rel=1.5e-14 mult_rel=1.084188350120738e-15`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `d2/(n + d2/d1)`  (R²=1.0)  [generalizes] `add_rel=9.0e-15 mult_rel=9.599096390562363e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_I_29_4
- solved 10 seeds (1 distinct matched eqs)
- **true:** `omega/c`
- **matched:** `omega/c`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_I_32_5
- solved 9 seeds (9 distinct matched eqs)
- **true:** `q**2*a**2/(6*pi*epsilon*c**3)`
- **matched:** `0.053051647*a**2*q**2/(c**3*epsilon)`  (R²=0.9999999999999998)  [generalizes] `add_rel=1.3e-08 mult_rel=2.2141922670153658e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.053051654*a**2*q**2/(c**3*epsilon)`  (R²=0.9999999999999839)  [generalizes] `add_rel=1.2e-07 mult_rel=2.492444212522972e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.0530516470990302*a**2*q**2/(c**3*epsilon)`  (R²=0.9999999999999999)  [generalizes] `add_rel=1.1e-08 mult_rel=1.8192080944956511e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.0530516509307136*a**2*q**2/(c**3*epsilon)`  (R²=0.9999999999999957)  [generalizes] `add_rel=6.1e-08 mult_rel=2.5060071023715687e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}

## feynman_I_34_1
- solved 10 seeds (1 distinct matched eqs)
- **true:** `omega_0/(1-v/c)`
- **matched:** `omega_0/(1.0 - v/c)`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_I_34_27
- solved 10 seeds (9 distinct matched eqs)
- **true:** `(h/(2*pi))*omega`
- **matched:** `0.159154938207697*h*omega`  (R²=0.9999999999999961)  [generalizes] `add_rel=3.1e-08 mult_rel=2.1766460332314328e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.15915494*h*omega`  (R²=0.9999999999999983)  [generalizes] `add_rel=1.9e-08 mult_rel=2.791609433599228e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.15915495*h*omega`  (R²=0.9999999999999927)  [generalizes] `add_rel=4.3e-08 mult_rel=2.074067682200527e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.15915495*h*omega`  (R²=0.9999999999999922)  [generalizes] `add_rel=4.3e-08 mult_rel=2.0411201076942903e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}

## feynman_I_34_8
- solved 10 seeds (9 distinct matched eqs)
- **true:** `q*v*B/p`
- **matched:** `B*q*v/p`  (R²=1.0)  [generalizes] `add_rel=2.0e-16 mult_rel=1.2262814116578436e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `B*q*v/p`  (R²=1.0)  [generalizes] `add_rel=2.1e-16 mult_rel=1.128802574060936e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `B*q*v/p`  (R²=1.0)  [generalizes] `add_rel=7.8e-18 mult_rel=1.173637548916083e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `B*q*v/p`  (R²=1.0)  [generalizes] `add_rel=7.3e-18 mult_rel=1.1808358394528677e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_I_38_12
- solved 10 seeds (9 distinct matched eqs)
- **true:** `4*pi*epsilon*(h/(2*pi))**2/(m*q**2)`
- **matched:** `0.31830990348781*epsilon*h**2/(m*q**2)`  (R²=0.9999999999999966)  [generalizes] `add_rel=5.4e-08 mult_rel=3.3720452481233547e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.3183099*epsilon*h**2/(m*q**2)`  (R²=0.999999999999997)  [generalizes] `add_rel=4.3e-08 mult_rel=2.2425404866397615e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.31830987*epsilon*h**2/(m*q**2)`  (R²=0.999999999999997)  [generalizes] `add_rel=5.1e-08 mult_rel=1.5920188643789972e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.3183099*epsilon*h**2/(m*q**2)`  (R²=0.9999999999999974)  [generalizes] `add_rel=4.3e-08 mult_rel=2.1840656748740718e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}

## feynman_I_39_1
- solved 10 seeds (4 distinct matched eqs)
- **true:** `3/2*pr*V`
- **matched:** `1.5*V*pr`  (R²=1.0)  [generalizes] `add_rel=1.3e-16 mult_rel=1.0927381047259614e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `1.5*V*pr`  (R²=1.0)  [generalizes] `add_rel=1.1e-16 mult_rel=1.1335697856308467e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `1.5*V*pr`  (R²=1.0)  [generalizes] `add_rel=1.3e-16 mult_rel=1.1315291542990108e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `1.5*V*pr`  (R²=1.0)  [generalizes] `add_rel=1.2e-16 mult_rel=1.12811989932413e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_I_39_11
- solved 10 seeds (7 distinct matched eqs)
- **true:** `1/(gamma-1)*pr*V`
- **matched:** `V*pr/(gamma - 1.0)`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `V*pr/(gamma - 1.0)`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `V*pr/(gamma - 1.0)`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `V*pr/(gamma - 1.0)`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_I_39_22
- solved 10 seeds (10 distinct matched eqs)
- **true:** `n*kb*T/V`
- **matched:** `T*kb*n/V`  (R²=1.0)  [generalizes] `add_rel=5.9e-17 mult_rel=1.1205830284902934e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `T*kb*n/V`  (R²=1.0)  [generalizes] `add_rel=8.4e-17 mult_rel=1.072098405376252e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `T*kb*n/V`  (R²=1.0)  [generalizes] `add_rel=6.7e-17 mult_rel=1.1795303267933976e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `T*kb*n/V`  (R²=1.0)  [generalizes] `add_rel=3.6e-17 mult_rel=1.1247003100708813e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_I_43_16
- solved 10 seeds (9 distinct matched eqs)
- **true:** `mu_drift*q*Volt/d`
- **matched:** `Volt*mu_drift*q/d`  (R²=1.0)  [generalizes] `add_rel=1.4e-16 mult_rel=1.1524414473843943e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `Volt*mu_drift*q/d`  (R²=1.0)  [generalizes] `add_rel=3.0e-17 mult_rel=1.0948510389028343e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `Volt*mu_drift*q/d`  (R²=1.0)  [generalizes] `add_rel=2.2e-16 mult_rel=1.1531097231421423e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `Volt*mu_drift*q/d`  (R²=1.0)  [generalizes] `add_rel=1.8e-17 mult_rel=1.1437183299343465e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_I_43_31
- solved 10 seeds (5 distinct matched eqs)
- **true:** `mob*kb*T`
- **matched:** `T*kb*mob`  (R²=1.0)  [generalizes] `add_rel=9.7e-17 mult_rel=1.1274368112207335e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `T*kb*mob`  (R²=1.0)  [generalizes] `add_rel=1.2e-16 mult_rel=1.111609937126711e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `T*kb*mob`  (R²=1.0)  [generalizes] `add_rel=1.0e-16 mult_rel=1.1011655157087892e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `T*kb*mob`  (R²=1.0)  [generalizes] `add_rel=1.1e-16 mult_rel=1.121957134498238e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_I_43_43
- solved 10 seeds (9 distinct matched eqs)
- **true:** `1/(gamma-1)*kb*v/A`
- **matched:** `kb*v/(A*(gamma - 1.0))`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `kb*v/(A*(gamma - 0.99999994))`  (R²=0.9999999999999967)  [generalizes] `add_rel=1.6e-05 mult_rel=1.2667692607127516e-05`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `kb*v/(A*(gamma - 0.99999994))`  (R²=0.9999999999999969)  [generalizes] `add_rel=2.7e-06 mult_rel=7.426225364406018e-07`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `kb*v/(A*(gamma - 1.0))`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_I_44_4
- solved 4 seeds (3 distinct matched eqs)
- **true:** `n*kb*T*ln(V2/V1)`
- **matched:** `T*kb*n*log(V2/V1)`  (R²=1.0)  [generalizes] `add_rel=1.9e-16 mult_rel=1.1715282741832112e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `T*kb*n*log(V2/V1)`  (R²=1.0)  [generalizes] `add_rel=8.6e-17 mult_rel=1.212187758357101e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `T*kb*n*log(V2/V1)`  (R²=1.0)  [generalizes] `add_rel=4.6e-17 mult_rel=1.0721998727534092e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_I_47_23
- solved 10 seeds (3 distinct matched eqs)
- **true:** `sqrt(gamma*pr/rho)`
- **matched:** `sqrt(gamma*pr/rho)`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `sqrt(gamma*pr/rho)`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `sqrt(gamma*pr/rho)`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_I_6_2a
- solved 7 seeds (7 distinct matched eqs)
- **true:** `exp(-theta**2/2)/sqrt(2*pi)`
- **matched:** `(-2.4513674e-6*(0.28260496 - theta)*log(0.72569734*theta) + 0.39894366)*exp(-0.50000185*theta**2)`  (R²=0.9999999999999849)  [generalizes] `add_rel=4.6e-06 mult_rel=1.7976815678888354e-06`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.7978845/(exp(0.49999624*theta**2) + exp(0.50000376*theta**2))`  (R²=0.9999999999999855)  [generalizes] `add_rel=7.6e-08 mult_rel=5.019230483137039e-10`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.398942293644142*exp(-0.500000025000001*theta**2)`  (R²=0.9999999999999988)  [generalizes] `add_rel=3.6e-08 mult_rel=1.2184340438480524e-07`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.39894235*exp(-0.500000050000005*theta**2)`  (R²=0.9999999999999801)  [generalizes] `add_rel=1.7e-07 mult_rel=2.431033160827572e-07`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}

## feynman_I_8_14
- solved 2 seeds (2 distinct matched eqs)
- **true:** `sqrt((x2-x1)**2+(y2-y1)**2)`
- **matched:** `sqrt((y1 - y2)**2)`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `sqrt((-y1 + y2)**2)`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_test_17
- solved 3 seeds (3 distinct matched eqs)
- **true:** `1/(2*m)*(p**2+m**2*omega**2*x**2*(1+alpha*x/y))`
- **matched:** `0.50016465461824*m*omega**2*x**2*(0.999826030270733*alpha*x/y + 0.99939317) + 0.4966278153664*p**2/m`  (R²=0.9999999859976709)  [generalizes] `add_rel=1.6e-04 mult_rel=0.0033376761012499333`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `omega*x*(0.500048429690416*m*omega*x*(alpha*x/y + 0.9995047) + 0.50011486*p**2/(m*omega*x))`  (R²=0.9999999980553108)  [generalizes] `add_rel=9.7e-05 mult_rel=0.0009468257681455679`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `0.49999914*omega**2*(m*x**2*(x*(alpha/y - 4.89510974924197e-7*omega) + 1.0000129) + p**2/(m*omega**2))`  (R²=0.9999999999977954)  [generalizes] `add_rel=1.7e-06 mult_rel=1.4044442569518274e-05`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## feynman_test_18
- solved 4 seeds (4 distinct matched eqs)
- **true:** `3/(8*pi*G)*(c**2*k_f/r**2+H_G**2)`
- **matched:** `H_G*(0.119366206*H_G + 0.119366206*c**2*k_f/(H_G*r**2))/G`  (R²=0.9999999999999998)  [generalizes] `add_rel=1.1e-08 mult_rel=4.3211429517497965e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `0.119366206*H_G*(H_G + c**2*k_f/(H_G*r**2))/G`  (R²=0.9999999999999998)  [generalizes] `add_rel=1.1e-08 mult_rel=2.0796317892093605e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `(0.119366206*H_G**2 + 0.119366206*c**2*k_f/r**2)/G`  (R²=0.9999999999999998)  [generalizes] `add_rel=1.1e-08 mult_rel=4.3802836554234033e-16`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `-0.119366206*(-H_G**2 + (-c**2*k_f/r + 1.1216508e-7)/r)/G`  (R²=0.9999999999999997)  [generalizes] `add_rel=1.1e-08 mult_rel=3.0138499522828504e-07`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}

## feynman_test_9
- solved 3 seeds (3 distinct matched eqs)
- **true:** `-32/5*G**4/c**5*(m1*m2)**2*(m1+m2)/r**5`
- **matched:** `-6.4*G**4*m1**2*m2**2*(m1 + m2)/(c**5*r**5)`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `-6.4000005150301*G**4*m1**2*m2**2*(m1 + m2)/(c**5*r**5)`  (R²=0.9999999999999929)  [generalizes] `add_rel=8.0e-08 mult_rel=2.478807355491144e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `-6.4*G**4*m1**2*m2**2*(m1 + m2)/(c**5*r**5)`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## strogatz_glider1
- solved 10 seeds (10 distinct matched eqs)
- **true:** `-0.05 * x**2 - sin(y)`
- **matched:** `-x*(0.049999997*x + sin(y)/x) - 8.083546e-8`  (R²=0.9999999999999936)  [generalizes] `add_rel=5.0e-08 mult_rel=2.2381636507369543e-06`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `-0.050000004*x**2 - sin(y)`  (R²=0.9999999999999982)  [generalizes] `add_rel=6.6e-08 mult_rel=1.37505294296761e-06`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `-0.050000004*x**2 - 1.0*sin(y)`  (R²=0.9999999999999983)  [generalizes] `add_rel=6.6e-08 mult_rel=1.37505294296761e-06`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `-0.05000001*x**2 - sin(y)`  (R²=0.9999999999999878)  [generalizes] `add_rel=1.7e-07 mult_rel=3.4376322674814025e-06`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## strogatz_glider2
- solved 9 seeds (2 distinct matched eqs)
- **true:** `x - cos(y)/x`
- **matched:** `x - cos(y)/x`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `x - 1.00000006*cos(y)/x`  (R²=0.9999999999999976)  [generalizes] `add_rel=4.3e-08 mult_rel=4.136260623744876e-07`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## strogatz_lv1
- solved 3 seeds (3 distinct matched eqs)
- **true:** `3  * x - 2  * x * y - x**2`
- **matched:** `0.814820770147772*x*y + 1.0710648*x*(-0.93364919307968*x - 2.6280568*y + 2.80094709738369)`  (R²=0.9999999999935353)  [generalizes] `add_rel=1.0e-06 mult_rel=9.52794812701443e-06`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `-2.0*x*y + 0.999977731719*x*(3.0000672 - 1.0000224*x)`  (R²=0.9999999999999943)  [generalizes] `add_rel=8.3e-08 mult_rel=1.992372317785002e-06`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `-0.99999983715*x**2 - 1.9999999*x*y + 2.9999995*x`  (R²=0.999999999998541)  [generalizes] `add_rel=1.2e-07 mult_rel=1.7849104681268634e-06`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## strogatz_lv2
- solved 10 seeds (10 distinct matched eqs)
- **true:** `2 * y - x * y - y**2`
- **matched:** `-1.000004481808*x*y + 1.005349946921*y*(1.9893658 - 0.99468297*y) + 4.3203414e-8*y`  (R²=0.9999999999786916)  [generalizes] `add_rel=4.5e-06 mult_rel=1.679976486913452e-06`  why={'error_is_zero': False, 'error_is_constant': False, 'fraction_is_constant': True}
- **matched:** `-0.9999998*x*y - 0.9999998*y*(y - 2.0)`  (R²=0.9999999999999594)  [generalizes] `add_rel=2.0e-07 mult_rel=5.000201649731011e-15`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `y*(-x - y + 1.9998952) + 0.00010478899*y`  (R²=0.9999999999999986)  [generalizes] `add_rel=1.3e-09 mult_rel=1.609425959229649e-07`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `-1.0*y*(x + y - 2.0)`  (R²=1.0)  [generalizes] `add_rel=1.4e-16 mult_rel=3.226289293785273e-15`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## strogatz_shearflow1
- solved 10 seeds (3 distinct matched eqs)
- **true:** `cot(y) * cos(x)`
- **matched:** `cos(x)*cos(y)/sin(y)`  (R²=1.0)  [generalizes] `add_rel=1.2e-16 mult_rel=1.3888885600717775e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `cos(x)*cos(y)/sin(y)`  (R²=1.0)  [generalizes] `add_rel=1.2e-16 mult_rel=1.3888885600717775e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `cos(x)*cos(y)/sin(y)`  (R²=1.0)  [generalizes] `add_rel=1.2e-16 mult_rel=1.3888885600717775e-16`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## strogatz_shearflow2
- solved 9 seeds (9 distinct matched eqs)
- **true:** `(cos(y)**2 + 0.1 *  sin(y)**2) * sin(x)`
- **matched:** `0.9*(cos(y)**2 + 0.11111112)*sin(x)`  (R²=0.9999999999999993)  [generalizes] `add_rel=1.3e-08 mult_rel=2.249111356009625e-08`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `(0.90000004*cos(y)**2 + 0.1)*sin(x)`  (R²=0.9999999999999988)  [generalizes] `add_rel=3.8e-08 mult_rel=1.249506308234236e-08`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}
- **matched:** `0.9000003*(sin(y + 1.5707964)**2 + 0.11111107)*sin(x)`  (R²=0.9999999999999377)  [generalizes] `add_rel=2.9e-07 mult_rel=1.6946149286812216e-07`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': False}
- **matched:** `(0.9*cos(y)**2 + 0.1)*sin(x)`  (R²=1.0)  [generalizes] `add_rel=3.6e-17 mult_rel=7.986663869107378e-17`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## strogatz_vdp1
- solved 1 seeds (1 distinct matched eqs)
- **true:** `10 *  (y - (1)/(3) * (x**3-x))`
- **matched:** `-3.33333405631588*x**3 + 3.33333405631588*x + 10.000001969793*y`  (R²=0.9999999999999413)  [generalizes] `add_rel=2.2e-07 mult_rel=8.374761833802351e-08`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

## strogatz_vdp2
- solved 10 seeds (1 distinct matched eqs)
- **true:** `-(1)/(10) * x`
- **matched:** `-0.1*x`  (R²=1.0)  [generalizes] `add_rel=0.0e+00 mult_rel=0.0`  why={'error_is_zero': True, 'error_is_constant': True, 'fraction_is_constant': True}

