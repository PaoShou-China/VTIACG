import numpy as np

np.random.seed(0)
dt = 0.005
g = 9.8


def normalize_angle(angle):
    """将角度归一化到 [-π, π] 范围内。"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def process_angle(current, new_temp):
    """归一化 new_temp 到 [-π, π] 并根据条件更新当前角度。"""
    new_temp = normalize_angle(new_temp)

    # 根据条件更新当前角度
    if abs(current - new_temp) > np.pi / 2:
        current = new_temp + np.sign(current - new_temp) * 2 * np.pi
    else:
        current = new_temp

    return current


class Missile:
    """导弹类，用于模拟导弹的运动和制导。"""

    def __init__(self, position, velocity, N=3):
        """初始化导弹的位置、速度和制导系数。"""
        self.pos = np.array(position)  # (x, y) 位置坐标
        self.vel = np.array(velocity)  # (vx, vy) 速度向量
        self.N = N  # 制导系数

    def calculate_bearing(self, target_pos):
        """计算导弹与目标之间的方位角。"""
        dx = target_pos[0] - self.pos[0]
        dy = target_pos[1] - self.pos[1]
        return np.arctan2(dy, dx)

    def run(self, target_pos, target_vel):
        """更新导弹的位置和速度，并返回当前状态。"""
        bearing_to_target = self.calculate_bearing(target_pos)
        R = np.linalg.norm(target_pos - self.pos)
        bearing_rate = ((target_vel[0] - self.vel[0]) * np.sin(bearing_to_target) -
                        (target_vel[1] - self.vel[1]) * np.cos(bearing_to_target)) / (R + 1e-8)
        acc = self.N * bearing_rate * np.linalg.norm(self.vel)
        acc_x = acc * self.vel[1] / np.linalg.norm(self.vel)
        acc_y = -acc * self.vel[0] / np.linalg.norm(self.vel)
        self.vel += np.array([acc_x, acc_y]) * dt
        self.pos += self.vel * dt
        Theta = np.arctan2(self.vel[1], self.vel[0])
        return np.copy(self.pos), np.copy(self.vel), -acc + g * np.cos(Theta)


class FakeTarget:
    """假目标类，用于模拟假目标的运动。"""

    def __init__(self, attack_angle, k1, k2, k3):
        """初始化假目标的攻击角度和参数。"""
        self.attack_angle = normalize_angle(attack_angle)
        self.k1, self.k2, self.k3 = k1, k2, k3
        self.pos = np.array([0.0, 0.0])

    def run(self, missile_pos, missile_vel):
        """更新假目标的位置并返回当前状态。"""
        R = np.linalg.norm(missile_pos)
        theta = np.arctan2(-missile_vel[1], -missile_vel[0])
        d_theta = normalize_angle(self.attack_angle - theta)
        beta = self.attack_angle + self.k1 * d_theta
        L = R * self.k2 * (1 + self.k3 * d_theta ** 2)
        pos = np.array([L * np.cos(beta), L * np.sin(beta)])
        self.pos = pos
        return np.copy(self.pos), np.array([0.0, 0.0])


def our_guidance_data(parameter, k1=0.15, k2=0.4, k3=0.1, N=4.0):
    """模拟导弹的制导过程并返回制导数据。"""
    x = parameter[0] * 600 ** 2 / 9.8
    y = parameter[1] * 600 ** 2 / 9.8
    theta = parameter[2]
    theta_f = parameter[3]
    V = 600
    t = 0
    V0 = V  # 标称速度
    r0 = V0 ** 2 / g  # 标称距离

    data = {
        'x': np.array(x),
        'y': np.array(y),
        't': np.array(t),
        'a': np.empty(1),
        'e_theta': np.empty(1),
        'e_lambda': np.empty(1),
        'epsilon': np.empty(1),
        'total_a': 0
    }

    missile = Missile(position=[x, y], velocity=[V * np.cos(theta), V * np.sin(theta)], N=N)
    fake_target = FakeTarget(attack_angle=theta_f + np.pi, k1=k1, k2=k2, k3=k3)

    while t < 500:
        # 更新假目标的位置
        fake_target_pos, fake_target_vel = fake_target.run(missile.pos, missile.vel)
        # 更新导弹的位置和加速度
        missile_pos, missile_vel, missile_acc = missile.run(fake_target_pos, fake_target_vel)

        # 记录位置和加速度
        x, y = missile_pos[0], missile_pos[1]
        dxdt, dydt = missile_vel[0], missile_vel[1]
        a = missile_acc

        if np.linalg.norm(a) > 50:
            a = a / np.linalg.norm(a) * 50

        R_tm = 0 - np.array([x, y, 0])
        V_tm = 0 - np.array([dxdt, dydt, 0])

        left_time = -np.linalg.norm(R_tm) ** 2 / np.dot(R_tm, V_tm)
        if left_time < dt and t > 10 and np.linalg.norm(R_tm) < 100:
            angle_RV = np.arccos(np.clip(np.dot(R_tm, V_tm) / np.linalg.norm(R_tm) / np.linalg.norm(V_tm), -1, 1))
            miss_distance = np.linalg.norm(R_tm) * np.sin(angle_RV)  # 脱靶量
            print('Miss distance:', miss_distance)
            break

        Theta = np.arctan2(missile_vel[1], missile_vel[0])
        Lambda = np.arctan2(-y, -x)
        if t > 1e-8:
            e_theta1 = process_angle(e_theta1, Theta - theta_f)
            e_theta2 = process_angle(e_theta2, Lambda - theta_f)
        else:
            e_theta1 = Theta - theta_f
            e_theta2 = Lambda - theta_f

        t = t + dt

        data['x'] = np.vstack((data['x'], x))
        data['y'] = np.vstack((data['y'], y))
        data['t'] = np.vstack((data['t'], t))
        data['a'] = np.vstack((data['a'], a))
        data['e_theta'] = np.vstack((data['e_theta'], e_theta1))
        data['e_lambda'] = np.vstack((data['e_lambda'], e_theta2))
        data['total_a'] = data['total_a'] + 0.5 * abs(a)**2 * dt

    print('Performance index:', data['total_a'])
    print('e_theta:', (np.rad2deg(e_theta1) + 180) % 360 - 180)
    print('e_lambda:', (np.rad2deg(e_theta2) + 180) % 360 - 180)
    return data


def guidance_data(parameter, guidance_law):
    """根据制导律模拟导弹的制导过程并返回制导数据。"""
    if guidance_law == 1:
        return our_guidance_data(parameter)

    x = parameter[0] * 600 ** 2 / 9.8
    y = parameter[1] * 600 ** 2 / 9.8
    theta = parameter[2]
    theta_f = parameter[3]
    theta_for4 = theta
    V = 600
    t = 0.0
    V0 = V  # 标称速度
    r0 = V0 ** 2 / g  # 标称距离

    data = {
        'x': np.array(x),
        'y': np.array(y),
        't': np.array(t),
        'a': np.empty(1),
        'e_theta': np.empty(1),
        'e_lambda': np.empty(1),
        'epsilon': np.empty(1),
        'total_a': 0
    }

    while t < 500:
        dxdt = V * np.cos(theta)
        dydt = V * np.sin(theta)

        X = x / r0
        Y = y / r0
        Theta = theta
        Lambda = np.arctan2(-y, -x)
        R_tm = 0 - np.array([x, y, 0])
        V_tm = 0 - np.array([dxdt, dydt, 0])
        K = 4  # 比例导引系数
        eL = R_tm / np.linalg.norm(R_tm)
        eV = V_tm / np.linalg.norm(V_tm)
        left_time = -np.linalg.norm(R_tm) ** 2 / np.dot(R_tm, V_tm)
        # if left_time < dt and t > 10 and np.linalg.norm(R_tm) < 100:
        #     angle_RV = np.arccos(np.clip(np.dot(R_tm, V_tm) / np.linalg.norm(R_tm) / np.linalg.norm(V_tm), -1, 1))
        #     miss_distance = np.linalg.norm(R_tm) * np.sin(angle_RV)  # 脱靶量
        #     print('Miss distance:', miss_distance)
        #     break

        C = 0.5 * np.linalg.norm(V_tm) / np.linalg.norm(R_tm)
        omega_LOS = -np.linalg.norm(V_tm) / np.linalg.norm(R_tm) * np.cross(eL, eV) - C * (Lambda - theta_f) * np.array(
            [0, 0, 1])
        a_pn = K * np.linalg.norm(V_tm) * np.cross(omega_LOS, eV)
        a_pn_g = np.cross(np.array([dxdt, dydt, 0]), a_pn)[2] / V + g * np.cos(theta)

        Lambda = np.arctan2(-Y, -X)
        temp_e_theta1 = normalize_angle(Theta - theta_f)
        temp_e_theta2 = normalize_angle(Lambda - theta_f)
        if t != 0 and abs(e_theta1 - temp_e_theta1) > np.pi / 2:
            e_theta1 = temp_e_theta1 + np.sign(e_theta1 - temp_e_theta1) * 2 * np.pi
        else:
            e_theta1 = temp_e_theta1
        if t != 0 and abs(e_theta2 - temp_e_theta2) > np.pi / 2:
            e_theta2 = temp_e_theta2 + np.sign(e_theta2 - temp_e_theta2) * 2 * np.pi
        else:
            e_theta2 = temp_e_theta2

        if guidance_law == 2:  # 线性最优角度约束制导律
            C = 0.5 * np.linalg.norm(V_tm) / np.linalg.norm(R_tm)
            omega_LOS = -np.linalg.norm(V_tm) / np.linalg.norm(R_tm) * np.cross(eL, eV) - C * (
                    Lambda - theta_f) * np.array([0, 0, 1])
            a_pn = K * np.linalg.norm(V_tm) * np.cross(omega_LOS, eV)
            a = np.cross(np.array([dxdt, dydt, 0]), a_pn)[2] / V + g * np.cos(theta)
        elif guidance_law == 3:  # 常系数偏置比例导引
            N = 4
            C = 0.025  # N=N, C=N*C
            omega_LOS = -np.linalg.norm(V_tm) / np.linalg.norm(R_tm) * np.cross(eL, eV) - C * (
                    Lambda - theta_f) * np.array([0, 0, 1])
            a_pn = N * np.linalg.norm(V_tm) * np.cross(omega_LOS, eV)
            a = np.cross(np.array([dxdt, dydt, 0]), a_pn)[2] / V + g * np.cos(theta)
        else:  # 非线性最优角度约束制导律
            N = 4
            M = 2
            V = np.linalg.norm(V_tm)
            sigma_dot = np.linalg.norm(V_tm) / np.linalg.norm(R_tm) * np.cross(eL, eV)[2]
            r_dot = V * np.dot(eL, eV)
            r = np.linalg.norm(R_tm)
            gamma = theta_for4
            sigma = Lambda
            gamma_f = N / (N - 1) * sigma - 1 / (N - 1) * gamma
            epsilon = theta_f - gamma_f
            data['epsilon'] = np.vstack((data['epsilon'], Lambda))
            a = N * V * sigma_dot + M * (N - 1) * V * r_dot / r * epsilon + g * np.cos(theta)

        if np.linalg.norm(a) > 50:
            a = a / np.linalg.norm(a) * 50
        dthetadt = a / V - g * np.cos(theta) / V

        x = x + dxdt * dt
        y = y + dydt * dt
        theta = theta + dthetadt * dt
        theta_for4 = theta_for4 + dthetadt * dt
        theta = normalize_angle(theta)
        t = t + dt

        dxdt = V * np.cos(theta)
        dydt = V * np.sin(theta)

        X = x / r0
        Y = y / r0
        Theta = theta
        Lambda = np.arctan2(-y, -x)
        R_tm = 0 - np.array([x, y, 0])
        V_tm = 0 - np.array([dxdt, dydt, 0])
        K = 4  # 比例导引系数
        eL = R_tm / np.linalg.norm(R_tm)
        eV = V_tm / np.linalg.norm(V_tm)
        left_time = -np.linalg.norm(R_tm) ** 2 / np.dot(R_tm, V_tm)
        if left_time < dt and t > 10 and np.linalg.norm(R_tm) < 100:
            angle_RV = np.arccos(np.clip(np.dot(R_tm, V_tm) / np.linalg.norm(R_tm) / np.linalg.norm(V_tm), -1, 1))
            miss_distance = np.linalg.norm(R_tm) * np.sin(angle_RV)  # 脱靶量
            print('Miss distance:', miss_distance)
            break

        data['x'] = np.vstack((data['x'], x))
        data['y'] = np.vstack((data['y'], y))
        data['t'] = np.vstack((data['t'], t))
        data['a'] = np.vstack((data['a'], a))
        data['e_theta'] = np.vstack((data['e_theta'], e_theta1))
        data['e_lambda'] = np.vstack((data['e_lambda'], e_theta2))
        data['total_a'] = data['total_a'] + 0.5 * abs(a)**2 * dt

    print('Performance index:', data['total_a'])
    print('e_theta:', (np.rad2deg(e_theta1) + 180) % 360 - 180)
    print('e_lambda:', (np.rad2deg(e_theta2) + 180) % 360 - 180)
    return data


def random_parameters(n):
    parameter_list = []
    for _ in range(n):
        sigma0 = np.random.uniform(-np.pi, np.pi)
        rs = np.random.uniform(0.7, 1.3)
        lambda0 = np.random.uniform(-np.pi, np.pi)
        theta0 = lambda0 + sigma0
        if theta0 > np.pi:
            theta0 = theta0 - 2 * np.pi
        elif theta0 < -np.pi:
            theta0 = theta0 + 2 * np.pi
        theta_f = np.random.uniform(-np.pi, np.pi)
        parameter = [rs * np.cos(lambda0 + np.pi), rs * np.sin(lambda0 + np.pi), theta0, theta_f]
        parameter_list.append(parameter)
    return parameter_list


parameter_list = [[-0.9018117165600209, -0.6794359383679132, 0.952385625317643, 0.2820093559455552],
                  [-1.0049805857143703, 0.41563155705394983, -0.8718421602418251, 2.46158236226362],
                  [0.24104798215611292, -0.8982853719845996, -1.5369437898957887, 0.1815521352435825],
                  [1.132378326259571, 0.5418883571715956, -2.267723349941997, -2.594143117880226],
                  [0.21111585884741854, -1.1808483624053865, -1.266846305197109, 2.324854893342369],
                  [-1.1451162580622614, 0.2826969001200206, 2.7652154305479875, 1.7626167986782493],
                  [0.6731882495632129, 0.8495709784581906, 1.6438554040191953, 2.7939372061654035],
                  [-0.086651909018129, 0.9448319983294212, -1.3420636644755166, 1.7230610881867197],
                  [1.0338135823550103, 0.12262205963639197, 2.984136863274648, 0.7391256268299178],
                  [1.0040107416284239, -0.37041288909740805, -2.7907157021601305, 1.1424106318740046],
                  [-0.31092902395712085, -0.9105980878321237, 0.35901553160091915, -2.763184855130804],
                  [0.27158409381394444, 1.068405234565175, -0.7718938607490056, -2.331524834455757],
                  [-0.8303521450740963, -0.39198871377443234, -0.7186385570939946, -0.38577806849661256],
                  [0.1945082834559151, 0.7359571321208819, 1.2393620350212393, -2.1280550609065623],
                  [-0.8329590801939103, 0.1789986195089598, 0.7503323239715693, -1.6058213652266506],
                  [-0.42537320118959626, -0.6373056724453864, -1.1605095217341845, -2.2733635639734713],
                  [0.3974371015897689, -0.8310943774130337, 0.11043069845364828, -2.5314873442184376],
                  [0.7493864013711635, -0.11165683817311836, -1.1661317197081225, -0.1969703092048971],
                  [-0.0716482089853031, -1.0604897395670891, -1.7842696362085038, -2.8953684930766395],
                  [-0.220720336711262, 0.7398977234584646, -2.6455530172150006, -2.3956043943015395],
                  [0.8725486727821831, 0.3720493345285108, 2.400997847249519, 1.2093379924674243],
                  [-0.8500832565455311, -0.12506397888470508, 0.5645411064700401, -2.5513470166434065],
                  [-0.5251976600980341, 1.1426588012220575, -0.6627789872136787, 1.051870439657157],
                  [-0.276883446267129, 1.0953425434322126, 2.6465025035910577, -1.99056737942446],
                  [0.33887637701723156, -0.6262576887560705, 2.6103679594014593, -3.11209010656657],
                  [-0.08007556981375737, -0.8582774321930312, 2.59502187550879, 2.904016275828597],
                  [-0.8756478194482775, -0.5715922286453059, -1.0003140468988059, 0.4539721128809391],
                  [-1.2021159947705686, 0.41474009838910736, -2.0721504612730333, 2.1765498811501764],
                  [0.3427797580171007, -0.8088249640783165, -3.058170205238218, -0.6502736084876624],
                  [0.7722982530323285, -0.7095496855497706, -1.4901292817342764, 1.209711657944542],
                  [0.962935542044563, -0.2726628533828584, -2.0022129332994805, 0.9047171041775188],
                  [1.0561096082752155, 0.12798194414357664, 2.783754235089633, -1.2467421964448875],
                  [-0.6445635804211196, -0.5903348095828815, 1.747912826879034, -0.44755945162887745],
                  [-0.7953977792637725, -0.37407161225801555, -1.8507815048270708, 0.5709703982973604],
                  [-0.6300849544116983, -0.891786583710661, 1.422692342772721, -0.43091067882464307],
                  [-0.8468015069427439, 0.36099279892049035, 2.088603185417602, 2.4625270657749994],
                  [0.907045230649997, 0.6609846060407518, -0.5879749747237257, 2.6356869952978226],
                  [0.7673533584683025, 1.0485087658551437, -0.8564634760617507, 2.3130042348459128],
                  [0.7617199009937025, 0.7505076376785561, 1.7989571166539298, 2.186600193255047],
                  [-0.8693207850122903, 0.5735165450367382, 1.3477574229482636, -2.7070036040043126],
                  [-0.16981002545512303, -0.9571795840428291, 2.6356980466054596, 2.3020480470850453],
                  [1.2101966398862967, 0.08923379586144427, -0.08020115846946752, -0.8797837679523912],
                  [-0.7959737076400147, -0.105826340045804, 1.577250217627519, -2.800177003634899],
                  [0.1927999516042422, -0.6844778933475737, -0.039620332386021806, -1.7346323436540405],
                  [-0.35508810174213856, -1.2056455888582416, 0.3126895191166059, -2.9415427593623535],
                  [-0.949035448389051, -0.5004181453862827, -1.6215472160918503, -1.6468679736459828],
                  [-1.041714644893002, -0.23720352412525764, 2.952134516352814, 0.5649210423999045],
                  [-0.7118598588955442, 0.5294533240709689, 0.8064034282539176, -1.8231054932258244],
                  [-0.0830995336142598, -1.2638945331975389, -0.4665654589659902, -0.05994907350709111],
                  [0.7965661427977239, 0.3040276697443615, 1.7934962166546402, -0.4120724946884726],
                  [-0.8039558512636634, 0.7766243953247612, -1.9506291961925606, -2.0131094656316755],
                  [-0.31777914061702395, -0.6686809499853676, -1.8593735913011114, -0.2909313059615086],
                  [1.2357226075784753, -0.07510332927301266, -2.972460875572166, -1.7787887080457416],
                  [0.850780915724822, 0.11101600141520401, -1.9871880275283518, 1.6234409614720846],
                  [-0.7905167260440208, -0.49002960918406585, -0.5759528032440633, 2.0800387899045303],
                  [-0.18033293004849335, 1.2102286911319422, -0.6124605349953423, 1.8726834876908143],
                  [-0.48673549347460227, -1.1748386485908577, -0.7971840323738801, -1.7875179831654804],
                  [-0.028193623259460492, 1.1381643435808, 1.264882065068587, -1.8013139715490059],
                  [0.1889038942311163, 0.690006586160876, -1.7236612655494519, -0.47321515615765364],
                  [-0.16895081167880768, 0.9634436996350738, -2.187813376113569, 0.5452821305591491],
                  [-0.7659299187042661, -0.08397061403551197, 2.3953683477944026, -2.3117842682548946],
                  [-0.8595308413357559, -0.3746567320873642, 1.773623788034751, -1.9900114795872133],
                  [-0.6115305562055849, 0.7821440215721979, -3.1386992643906466, 2.7673155272249987],
                  [0.9452527327638094, -0.6535699729367858, -2.0794516338459523, -2.6174340329313646],
                  [1.02078009774255, -0.24889417237266095, -3.052818480478012, -1.3059756057179956],
                  [0.7561295448527807, 0.07833401465264338, 1.6164022587622755, 2.698812292271578],
                  [-0.23193244897735532, 1.147895125655576, -0.3038130435397277, 0.542931086572874],
                  [0.9814819795124914, -0.1397161507245728, 0.260440581110287, 2.3656522254492494],
                  [0.14648944909409056, 1.2685117064799813, -2.7026456944135835, 2.8231534232414432],
                  [-0.8050148924509627, -0.8621033090849649, -2.690298837777752, 2.3517206525519905],
                  [-0.8925558908797981, -0.816033348434525, -0.5598508085497025, -3.0584230234058496],
                  [0.7837486893346839, -0.08987086540314827, 2.0675632271155617, -0.13590336900515254],
                  [-0.7347778095745559, 0.7965372056936821, -0.8420978058742095, -2.2814228779859445],
                  [-0.8118512597788438, -0.05783580784602642, 2.095044673216192, -1.7321671929855338],
                  [1.1997357413578595, -0.20613040076545397, 0.4446231586806926, 2.895509552580571],
                  [-0.5810214540736764, 1.0091122671664452, 1.5060836100542523, -2.6320175916253703],
                  [0.5649348800602751, 0.6207584325317128, -2.8919692023794146, -2.8058997699828],
                  [0.09115095805670054, -0.7009547626931407, 3.117560177988728, -2.218299650270828],
                  [-0.3545988712570963, -0.6651439863756847, -1.5609324173857884, -1.5999050057787487],
                  [0.6621212112594147, -0.7947470511725943, 1.7661445824311701, 1.4265611755472394],
                  [0.7322197638125022, 0.26556232871080526, 2.046447438782053, -1.2465925447537087],
                  [-0.39632369043848226, -0.8893754699308991, -0.34306515494850354, 1.2291509257485265],
                  [0.38902221520967856, 0.842475148645043, 2.9196065123203336, 1.8129849233769244],
                  [0.2005189424878275, -1.1000725851245268, -1.0333108380472975, 1.7430031167085982],
                  [-0.7877824852352784, -0.48343235945955837, -0.961187868163965, -1.4274020847995992],
                  [-0.7923417039028731, 0.20420372125512826, -1.0636887157714972, -2.861285297830474],
                  [-0.7409547040585028, -0.08809976223667226, 2.002017827483799, -1.2138479440331107],
                  [-0.7782444993126587, -1.0107640408971175, 1.4018615379011807, -2.9194039168248107],
                  [-0.9801316610577774, -0.22671186826664794, -0.20998446632719547, 1.1397227574584514],
                  [-0.6071774009650667, 0.485341421134281, -2.0717435432134583, 2.8676817315841845],
                  [-1.1956264931083647, -0.33765533876052245, -1.6905736821285726, -0.27073352242258064],
                  [-0.1575841014681392, -0.9623455266171473, -2.474261530986154, -0.6344426150722549],
                  [-0.3467645749279489, -1.0586707593921896, -2.4902371560001395, -1.0824646416631096],
                  [0.06777901708206598, 1.0795109129931661, -0.02011300322323617, -2.1328974829185183],
                  [-1.231633984589749, 0.3316291311228422, 1.5992610523595858, 0.5716703707477753],
                  [0.9301285112563811, -0.29015140260230654, -1.196336571612303, 0.47595858836212246],
                  [0.49832957991525995, -1.1412516106746866, -2.285251425210917, -2.139962039095396],
                  [0.867097216546763, 0.36052342966364465, -1.9376627764359897, -0.47731943874526506],
                  [1.1830395264870524, 0.2512397969146868, 1.8346190684602925, 2.883873494261872],
                  [0.909217963938613, 0.09360989278539858, 2.3354433354986726, -1.9777436292510917]]

if __name__ == '__main__':
    print(random_parameters(100))
