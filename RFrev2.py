"""
RFチャネルモデル（2024/5/1-）
VISAさんのコードを基に作成（2025/2-）
SAT研用に修正（2025/06）
対象: LEO衛星-HAPS間, LEO衛星-地上UE間, HAPS-地上UE間
参考:
[1]O. B. Yahia, E. Erdogan, G. K. Kurt, I. Altunbas and H. Yanikomeroglu, "HAPS Selection for Hybrid RF/FSO Satellite Networks," in IEEE Transactions on Aerospace and Electronic Systems, vol. 58, no. 4, pp. 2855-2867, Aug. 2022.
[2]M. R. Bhatnagar and M. K. Arti, "Performance Analysis of Hybrid Satellite-Terrestrial FSO Cooperative System," in IEEE Photonics Technology Letters, vol. 25, no. 22, pp. 2197-2200, Nov.15, 2013.
[3]Rec. ITU-R P.838-3, "Specific attenuation model for rain for use in prediction methods," March, 2005.
[4]Rec. ITU-R P.840-9, "Attenuation due to clouds and fog," March, 2005.
[5]G. A. Siles, J. M. Riera and P. Garcia-del-Pino, "Atmospheric Attenuation in Wireless Communication Systems at Millimeter and THz Frequencies [Wireless Corner]," in IEEE Antennas and Propagation Magazine, vol. 57, no. 1, pp. 48-61, Feb. 2015, doi: 10.1109/MAP.2015.2401796.
[6]トランジスタ技術編集部, "RFワールドNo.15," CQ出版, 2011.
[7]S. R, S. Sharma, N. Vishwakarma and A. S. Madhukumar, "HAPS-Based Relaying for Integrated Space–Air–Ground Networks With Hybrid FSO/RF Communication: A Performance Analysis," in IEEE Transactions on Aerospace and Electronic Systems, vol. 57, no. 3, pp. 1581-1599, June 2021, doi: 10.1109/TAES.2021.3050663.
[8]M. Takahashi, Y. Kawamoto and N. Kato, "Transmit power control of HAPS for two-way relay communication in space-air-ground integrated networks," 39th International Communications Satellite Systems Conference (ICSSC 2022), Stresa, Italy, 2022, pp. 122-127, doi: 10.1049/icp.2023.1372.
[9]K. Mashiko, Y. Kawamoto and N. Kato, "Combined Control of Coverage Area and HAPS Deployment in Hybrid FSO/RF SAGINs," 
"""
import numpy as np
from scipy.special import hyp1f1
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Parameters-----------------------------------------------------------------------------------------
# Instrumental Parameters
GT = 40         # dB 
GR = 40         # dB 
PT_S1 = 47.7815 # dBm (=60 W) [9]
#PT_H1 = 20     # transmit power of HAPS

# Climate Parameters
atm_att = 0.5  # dB [5]Fig.2(a)
L_rain = 5     # km through rainfall 
L_cloud = 5    # km through cloud 

#oxy = 0.005   # dB/km

# Other Parameters


BandWidth1 = 200e6
BandWidth2 = 1000e6
temperature = 273.15
sig_H = 10**(-171.6/10) # dBm/Hz for HAPS
sig_U = 10**(-175.6/10) # dBm/Hz for UE (Gateway)
C_LEO = 10e9    # bps (10Gbps)
C_TN  = 1e12    # bps (1Tbps)



# Shadowed-Rician Fading Parameters
'''
# Shadowed-Rician land mobile satellite (LMS) model
b1 = 0.126 #[2] LMS Channel Parameter Fig.1 Average Power of the multipath Component
m1 = 10.1 #[2] LM Channel Parameter Nakagami Parameter
Omega1 = 0.835 #[2] Average Power of LOS Component
'''
# Based on The lowpassequivalent complex envelope of the shadowed-Rician fading channel mode
b1 = 0.251 #[7] Moderate Shadowing
m1 = 5
Omega1 = 0.279

x_values = np.linspace(0.01, 1, 100) #pdf_h1_squaredで使用 #PDFの定義域x>0


# R = 30 # 25mm/h for HAPS


# Satellite-to-UE-----------------------------------------------------------------------------------------


# Cloud Attenuation
def complex_dielectric_permittivity(frequency, temperature): 
    eps0 = 77.66 + 103.3 * (300 / temperature - 1)
    eps1 = 0.0671 * eps0
    eps2 = 3.52
    fp = 20.20 - 146 * (300 / temperature - 1) + 316 * (300 / temperature - 1)**2
    fs = 39.8 * fp
    
    eps_prime = eps0 - (eps0 - eps1) / (1 + (frequency / fp)**2) - (eps1 - eps2) / (1 + (frequency / fs)**2) + eps2
    eps_double_prime = (frequency * (eps0 - eps1) / fp) / (1 + (frequency / fp)**2) + (frequency * (eps1 - eps2) / fs) / (1 + (frequency / fs)**2)
    return eps_prime, eps_double_prime

def specific_attenuation_coefficient(frequency, temperature, w):
    eps_prime, eps_double_prime = complex_dielectric_permittivity(frequency, temperature)
    eta = (2 + eps_prime) / eps_double_prime
    kl = 0.819 * frequency / (eps_double_prime * (1 + eta**2)) * w
    return kl



# Shadowed-Rician Fading Channel
def pdf_h1_squared(x): # PDF of the shadowed Rician fading channel [2] （ビザさんのコードにあったが今回未使用）
    alpha1 = 0.5 * (2*b1*m1 / (2*b1*m1 + Omega1))**m1 /b1
    beta1 = 0.5 / b1
    delta1 = 0.5 * Omega1 / (2*b1**2 * m1 + b1 * Omega1)
    PDF = alpha1 * np.exp(-beta1 * x) * hyp1f1(m1, 1, delta1 * x) # [2](2) PDF f(x) x>0
    h1_squared = np.trapz(x_values * PDF, x_values) #確立密度関数の期待値（平均）の計算
    #f_|h1|^2 (x)を|h_kl|^2(x)に変更する式ならよい
    #[2]のPDFを[1](15)[7](12)[8]のSNR式に代入するやり方
    return h1_squared

def shadowed_rician_pdf_fixed(h_prime): # the shadowed-Rician pdf of envelope of fading channel gain[7]
    if h_prime <= 0:
        return 0  # h' only h' > 0

    coeff = ((2 * b1 * m1) / (2 * b1 * m1 + Omega1)) ** m1  # 係数修正(*Omega1 -> +Omega1)
    power_term = h_prime / b1
    exp_term = np.exp(- (h_prime**2) / (2 * b1))
    hyper_term = (Omega1 * h_prime**2) / (2 * b1 * (2 * b1 * m1 + Omega1))
    
    # 超幾何関数の値が適切かチェック
    hypergeom_value = hyp1f1(m1, 1, hyper_term)
    if np.isnan(hypergeom_value) or np.isinf(hypergeom_value):
        return 0  # 数値が異常な場合は 0 を返す
    return coeff * power_term * exp_term * hypergeom_value

def shadowed_rician_pdf_multiplied(h_prime):
    return h_prime * shadowed_rician_pdf_fixed(h_prime)

def expectation_and_square():
    #integral_result, error_fixed = quad(shadowed_rician_pdf_fixed, 0, np.inf)
    expectation_result, error_fixed = quad(shadowed_rician_pdf_multiplied, 0, np.inf)
    #print(integral_result) #[7]-> 1.0000000000000002, [2]-> 1.0000000000000002
    #print(expectation_result) #[7]-> 0.7895078378003142, [2]-> 0.9785696225140601
    return expectation_result**2



# Distance and Angle
def distance_and_angle(pos1, pos2): # this function was used in Satellite to HAPS 
    if pos1[0] == 999:
        return 867.88e3, 0.6331 #cross path (36.27degree)
    else:
        distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)
        # elevation_angle = np.arccos(550 / distance)
        elevation_angle = 0
        return distance, np.degrees(elevation_angle)



# Attenuation
def free_space_loss(distance, freq): # Free Space Loss
    #print(20*np.log10(freq))
    return 20*np.log10(distance) + 20*np.log10(freq) + 20*np.log10(4*np.pi/3e8)

def rain_attenuation(rain_rate, pos1, pos2): # Rain Attenuation
    kh = 0.08084   # at 19GHz [3]
    kv = 0.08642
    alpha_h = 1.0691
    alpha_v = 0.9930
    L, angle = distance_and_angle(pos1, pos2)
    # print(f'angle1:{angle}')
    k = (kh+kv+(kh-kv)*np.cos(angle)**2*np.cos(np.pi))/2
    alpha = (kh*alpha_h+kv*alpha_v+(kh*alpha_h-kv*alpha_v)*np.cos(angle)**2*np.cos(np.pi))/(2*k)
    # print(f'k:{k}')
    # print(f'alpha:{alpha}')
    return k * rain_rate ** alpha

def rain_attenuation_for29GHz(rain_rate, pos1, pos2): # Rain Attenuation
    kh = 0.2224   # at 29GHz [3]
    kv = 0.2124
    alpha_h = 0.9580
    alpha_v = 0.9203
    L, angle = distance_and_angle(pos1, pos2)
    # print(f'angle1:{angle}')
    k = (kh+kv+(kh-kv)*np.cos(angle)**2*np.cos(np.pi))/2
    alpha = (kh*alpha_h+kv*alpha_v+(kh*alpha_h-kv*alpha_v)*np.cos(angle)**2*np.cos(np.pi))/(2*k)
    # print(f'k:{k}')
    # print(f'alpha:{alpha}')
    return k * rain_rate ** alpha


# Throughput Calculation
# input: pos1[km], pos2[km], rain_rate[mm/h], cloud_density[g/m^3], freq[GHz], cnt(separation for bandwidth)
# output: RF tuhroughput [bps]
def S2U_RF_throughput(pos1, pos2, rain_rate, cloud_density, freq, cnt): # Throughput Calculation
    BW = BandWidth1/cnt # Bandwidth (cnt: Separation)
    PT_S = PT_S1/cnt
    L, angle = distance_and_angle(pos1, pos2)
    L = L*1e3 # Convert to meters
    freq = freq*1e9 # Convert to Hz
    # print(f'angle2:{angle}')
    loss = free_space_loss(L, freq)
    kl = specific_attenuation_coefficient(freq, temperature, cloud_density)
    rain = rain_attenuation(rain_rate, pos1, pos2)
    h1_squared = expectation_and_square() # Phasing (Complete LOS)
    
    g_S2U = GT + GR - loss - atm_att - rain*L_rain - kl*L_cloud # Sum of Loss
    #↑Ptに対してどれだけ増減させますか？？
    # g_S2U = GT + GR - 92.45 - 20*np.log10(L) - 20*np.log10(2) - atm_att - rain*L_rain - kl*L_cloud # Sum of Loss
    SNR_S2U_RF = PT_S * 10 ** (g_S2U / 10) * h1_squared / (sig_U * BW) #[1](15)[7](12)[8] need to check the passing distance of the atmosphere
    C_S2U_RF = BW*np.log2(1+SNR_S2U_RF) # Throughput [bps]

    # print(f'L[m]      : {L}')
    # print(f'loss[dB]  : {loss:5f}')
    # print(f'rain[mm/h]: {rain_rate:5f}')
    # print(f'rain[dB]  : {rain*L_rain:5f}')
    # print(f'cloud[dB] : {kl*L_cloud:5f}')
    # print(f'oxy[dB]   : {atm_att}')
    # print(f'h1        : {h1_squared:5f}')
    # print(f'GAIN[dB]  : {g_S2U:5f}')
    # print(f'SNR       : {10*np.log10(SNR_S2U_RF):5f}')
    
    # print(f'92.45:{20*np.log10((4*np.pi*1e9)/3e8)}')
    # print(f'throughput[Mbps]: {C_S2U_RF/1e6:5f}')
    return C_S2U_RF


def S2U_RF_throughput_for29GHz(pos1, pos2, rain_rate, cloud_density, freq, cnt): # Throughput Calculation
    BW = BandWidth2/cnt # Bandwidth (cnt: Separation)
    PT_S = PT_S1/cnt
    L, angle = distance_and_angle(pos1, pos2)
    L = L*1e3 # Convert to meters
    freq = freq*1e9 # Convert to Hz
    # print(f'angle2:{angle}')
    loss = free_space_loss(L, freq)
    kl = specific_attenuation_coefficient(freq, temperature, cloud_density)
    rain = rain_attenuation_for29GHz(rain_rate, pos1, pos2)
    h1_squared = expectation_and_square() # Phasing (Complete LOS)
    g_S2U = GT + GR - loss - atm_att - rain*L_rain - kl*L_cloud # Sum of Loss
    #print(rain)

    SNR_S2U_RF = PT_S * 10 ** (g_S2U / 10) * h1_squared / (sig_U * BW) #[1](15)[7](12)[8] need to check the passing distance of the atmosphere
    C_S2U_RF = BW*np.log2(1+SNR_S2U_RF) # Throughput [bps]
    return C_S2U_RF



# Delay Summation
# input: pos1[km], pos2[km], data_size[MB], rain_rate[mm/h], cloud_density[g/m^3], freq[GHz]
# output: delay_sum [s]
def delay_sum(pos1, pos2, data_size, rain_rate, cloud_density, freq): # For Satellite-to-Gateway [s]
    pos1 = pos1
    pos2 = pos2
    data_size = data_size * 8 * 1e6 # Convert to bits
    freq = freq

    cnt = 1
    C_S2U_RF = S2U_RF_throughput(pos1, pos2, rain_rate, cloud_density, freq, cnt)
    # print(f'C_S2U_RF[Mbps]: {C_S2U_RF/1e6}')
    #print(data_size / C_S2U_RF)
    #print(distance_and_angle(pos1, pos2)[0] / (299792458))
    delay_sum = data_size / C_S2U_RF + distance_and_angle(pos1, pos2)[0] / (299792458) # [m/s]
    
    return delay_sum # [s]

def delay_hop(pos1, pos2, data_size, binary): # For among Satellites, Gateways
    cnt = 1
    if binary == 1:
        #print(distance_and_angle(pos1, pos2)[0] / (299792458))
        #print(data_size / C_LEO)
        return data_size / C_LEO + distance_and_angle(pos1, pos2)[0] / (299792458.0) # [m/s]
    else:
        #print(distance_and_angle(pos1, pos2)[0] / ((2/3) * (299792458)))
        #print(data_size / C_TN)
        return data_size / C_TN + distance_and_angle(pos1, pos2)[0] / ((2/3) * (299792458.0)) # [m/s]



# Test-----------------------------------------------------------------------------------------
# print(f'expectation_and_square:{expectation_and_square()}')

# Main-----------------------------------------------------------------------------------------
def main():
    freq = 19
    para_rain  = 0.0   # mm/h
    para_cloud = 0.5  # g/m^3
    para_packet     = 10    # MB
    # para_packet = packet * 8 * 1e6 # bit
    pos1 = np.array([0, 0, 1000]) 
    pos2 = np.array([0, 0, 0]) 

    #delay(pos1[m], pos2[m], data_size[bit], rain_rate[mm/h], cloud_density[g/m^3], freq[Hz]):
    result1 = delay_sum(np.array([0, 0, 0]), np.array([0, 0, 1000]), para_packet, para_rain, para_cloud, freq)
    print(f'S2U[ms]: {result1*1e3}')
    #999の時はななめパス指定
    result2 = delay_sum(np.array([999, 0, 0]), np.array([0, 0, 0]), para_packet, para_rain, para_cloud, freq)
    print(f'S9U[ms]: {result2*1e3}')
    result3 = S2U_RF_throughput(pos1, pos2, para_rain, para_cloud, 19, 1)
    print(f'S2U[Mbps]: {result3/1e6}')
    result4 = S2U_RF_throughput_for29GHz(pos1, pos2, para_rain, para_cloud, 29, 1)
    print(f'S2U[29GHz][Mbps]: {result4/1e6}')
    #delay_hop(pos1[m], pos2[m], data_size[bit], binary[1=constellation, 0=terrestrial]):
    #result = delay_hop(np.array([0, 0, 0]), np.array([0, 700e3, 0]), para_packet, 1)
    #print(f'S2S[ms]: {result*1e3}')
    #result = delay_hop(np.array([0, 0, 0]), np.array([0, 100e3, 0]), para_packet, 0)
    #print(f'U2U[ms]: {result*1e3}')
    #print(20*np.log10(4*np.pi/3e8))
    #print(f'expectation_and_square:{expectation_and_square()}')
if __name__ == "__main__":
    main()


'''Previous versions of this code
SNR_S2U_values = []
C_S2U_values = []


def S2U_RF(pos1, pos2, cnt):
    BW = 200e6/cnt
    PT_S = PT_S1/cnt
    L, angle = distance_and_angle(pos1, pos2)
    L_rain = 5 # (km)
    L_cloud = 5 # (km)
    w = 0.5 # (g/m^3)
    kl = specific_attenuation_coefficient(freq, temperature, w)
    k = (kh+kv+(kh-kv)*np.cos(angle)**2*np.cos(np.pi))/2
    alpha = (kh*alpha_h+kv*alpha_v+(kh*alpha_h-kv*alpha_v)*np.cos(angle)**2*np.cos(np.pi))/(2*k)
    rain = k * R ** alpha
    print(rain * L_rain)
    g_S2U = GT + GR - 92.45 - 20*np.log10(L) - 20*np.log10(2) - oxy*L - rain*L_rain - kl*L_cloud # Loss Calculation
    h1_squared_values = pdf_h1_squared(x_values)
    h1_squared = np.trapz(x_values * h1_squared_values, x_values) # フェージング（完全ロス）
    SNR_S2U_RF = PT_S * 10 ** (g_S2U / 10) * h1_squared / sig_U #[1] 大気の距離だけに変更必要
    print(f'h1:{h1_squared}')
    print(f'SNR: {10*np.log10(SNR_S2U_RF)}')

    # if SNR_S2U_RF >= 10:
    #     C_S2U_RF = BW*np.log2(1+SNR_S2U_RF)/1e6
    # else:
    #     C_S2U_RF = 0
    C_S2U_RF = BW*np.log2(1+SNR_S2U_RF)/1e6 # Throughput Calculation
    return C_S2U_RF
'''

# for x in x_values_pos1:
#     pos1 = np.array([x, 0, 550]) 
#     C_S2U = S2U_RF(pos1, pos2, 1)
#     #print(C_S2U)
#     C_S2U_values.append(C_S2U)

# pos1 = np.array([0, 0, 550])
# print(f'{S2U_RF_throughput(pos1, pos2, 1)}Mbps')

#plt.subplot(3, 2, 4)
#plt.plot(x_values_pos1, C_S2U_values)
#plt.xlabel('Distance (m)')
#plt.ylabel('C')
#plt.show()

# Satellite-to-HAPS-----------------------------------------------------------------------------------------
'''
def pdf_h1_squared(x): #[2]
    alpha1 = 0.5 * (2*b1*m1 / (2*b1*m1 + Omega1))**m1 /b1
    beta1 = 0.5 / b1
    delta1 = 0.5 * Omega1 / (2*b1**2 * m1 + b1 * Omega1)
    return alpha1 * np.exp(-beta1 * x) * hyp1f1(m1, 1, delta1 * x) # [2](2) PDF f(x)

def S2H_RF(pos1, pos2):
    L = np.linalg.norm(pos1 - pos2)
    g_S2H = GT+GR-20*np.log10(L)-20*np.log10(2)-oxy*L
    h1_squared_values = pdf_h1_squared(x_values)
    h1_squared = np.trapz(x_values*h1_squared_values, x_values)
    SNR_S2H_RF = PT_S1*10**(g_S2H/10)*h1_squared/sig_H #[1]
    if SNR_S2H_RF >= 10:
        C_S2H_RF = BW*np.log2(1+SNR_S2H_RF)/1e6
    else:
        C_S2H_RF = 0
    return C_S2H_RF

x_values_pos1 = np.linspace(0, 2000, 100)
pos2 = np.array([0, 0, 1])
"""
SNR_S2H_values = []
C_S2H_values = []

for x in x_values_pos1:
    pos1 = np.array([x, 0, 550e3]) 
    SNR_S2H, C_S2H = S2H_RF(pos1, pos2)
    SNR_S2H_values.append(SNR_S2H)
    C_S2H_values.append(C_S2H)

plt.figure(figsize=(12, 8))

plt.subplot(3, 2, 1)
plt.plot(x_values_pos1, SNR_S2H_values)
plt.xlabel('Distance (m)')
plt.ylabel('SNR (dB)')

plt.subplot(3, 2, 2)
plt.plot(x_values_pos1, C_S2H_values)
plt.xlabel('Distance (m)')
plt.ylabel('C')
"""
'''
# HAPS-to-UE-----------------------------------------------------------------------------------------
'''
def distance_and_angle2(position1, position2):
    distance = np.linalg.norm(position1 - position2)
    angle = np.arccos(19/distance)
    return distance, angle
def H2U_RF(pos1, pos2, cnt):
    BW = 60e6/cnt
    PT_H = PT_H1/cnt
    L, angle = distance_and_angle2(pos1, pos2)
    kl = specific_attenuation_coefficient(freq, temperature)
    k = (kh+kv+(kh-kv)*np.cos(angle)**2*np.cos(np.pi))/2
    alpha = (kh*alpha_h+kv*alpha_v+(kh*alpha_h-kv*alpha_v)*np.cos(angle)**2*np.cos(np.pi))/(2*k)
    rain = k*R**alpha
    g_H2U = GT+GR-92.45-20*np.log10(L)-20*np.log10(2)-oxy*L-rain*L-kl*L
    h1_squared_values = pdf_h1_squared(x_values)
    h1_squared = np.trapz(x_values*h1_squared_values, x_values)
    SNR_H2U_RF = PT_H*10**(g_H2U/10)*h1_squared/sig_U #[1]
    if SNR_H2U_RF >= 10:
        C_H2U_RF = BW*np.log2(1+SNR_H2U_RF)/1e6
    else:
        C_H2U_RF = 0
    return C_H2U_RF
"""
SNR_H2U_values = []
C_H2U_values = []

pos2 = np.array([0,0,1e3])

for x in x_values_pos1:
    pos1 = np.array([0, 0, 20e3]) 
    SNR_H2U, C_H2U = H2U_RF(pos1, pos2)
    SNR_H2U_values.append(SNR_H2U)
    C_H2U_values.append(C_H2U)

plt.subplot(3, 2, 5)
plt.plot(x_values_pos1, SNR_H2U_values)
plt.xlabel('Distance (m)')
plt.ylabel('SNR (dB)')

plt.subplot(3, 2, 6)
plt.plot(x_values_pos1, C_H2U_values)
plt.xlabel('Distance (m)')
plt.ylabel('C')

plt.tight_layout()
plt.show()
"""
'''