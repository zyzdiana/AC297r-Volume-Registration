def compute_RMS_from_C_output(output_path, output_filename):
    results = np.loadtxt(os.path.join(output_path,output_filename))
    splits = output_filename.split('_')
    res = splits[0][:-2]
    res_f = float('.'.join(res.split('_')))
    interp = splits[1]
    rot_ax = splits[2]
    trans_ax = splits[10]
    rang = '_'.join(splits[4:9])
    
    true_params = []
    Max_dis = []
    Max_dis_R = []
    RMS_ls = []
    RMS_rotation_ls = []
    RMS_translation_ls = []
    RMS_trans_R = []
    counter_ls = []    
    
    ref = rep_to_angle(0,rang)
    
    idx = 0
    for rep in xrange(1,36):
        rot_angle = rep_to_angle(rep,rang)
        true_params.append(rot_angle)
        true_t, true_RM = get_true_params(rot_angle,rot_ax)
        rad = res_to_rad(res)
        
        for algo in xrange(8):
            counter = results[idx,1]
            Ps = results[idx,2:]
            R_axis = Ps[3:]/np.linalg.norm(Ps[3:])
            t , RM = get_params(Ps, res_f)
            dt = true_t + t
            dR = RM.dot(true_RM.T)
            max_trans_R, E_Max = Max_displacement(dR, R_axis, dt)
            Max_dis.append(E_Max)
            Max_dis_R.append(max_trans_R)
            RMS_ls.append(RMS(dR, dt, res_f))
            
            if(RMS(dR, dt, res_f) > 1):
                print idx, rot_ax, rang, rep
            if(RMS(dR, dt, res_f) > E_Max):
                print rot_ax, rang, rep, rot_angle
                print RMS(dR, dt, res_f), E_Max

            RMS_rotation_ls.append(RMS_Rotation(dR))
            RMS_translation_ls.append(RMS_Translation(dt))
            RMS_trans_R.append(RMS_Translation_R(dR))
            counter_ls.append(counter)
            idx += 1
    return [true_params, 
            RMS_ls, RMS_rotation_ls, RMS_translation_ls, 
            Max_dis, Max_dis_R, RMS_trans_R, counter_ls]  


def compute_RMS(interp, algo, axes_dict):
    keys = axes_dict.keys()
    true_params = []
    Max_dis_6_4 = []
    Max_dis_R_6_4 = []
    RMS_ls_6_4 = []
    RMS_rotation_ls_6_4 = []
    RMS_translation_ls_6_4 = []
    RMS_trans_R_6_4 = []
    counter_6_4 = []
    
    Max_dis_8 = []
    Max_dis_R_8 = []
    RMS_ls_8 = []
    RMS_rotation_ls_8 = []
    RMS_translation_ls_8 = []
    RMS_trans_R_8 = []
    counter_8 = []
    
    Max_dis_10 = []
    Max_dis_R_10 = []
    RMS_ls_10 = []
    RMS_rotation_ls_10 = []
    RMS_translation_ls_10 = []
    RMS_trans_R_10 = []
    counter_10 = []
    for ix, rot_ax in enumerate(keys):
        for rang in ranges:
            ref = rep_to_angle(0,rang)
            for rep in xrange(1,36):
                rot_angle = rep_to_angle(rep,rang)
                true_params.append(rot_angle)
                true_t, true_RM = get_true_params(rot_angle,rot_ax)

                res = '6_4mm'
                res_f =  6.4
                rad = res_to_rad(res)
                output_file = '%s_%s_%s_rot_%s_deg_%s_trans.txt' % (res, interp, rot_ax, rang, axes_dict[rot_ax])
                #for algo in xrange(8):
                idx = algo+(rep-1)*8
                counter = results[idx,1]
                Ps = results[idx,2:]
                R_axis = Ps[3:]/np.linalg.norm(Ps[3:])
                t , RM = get_params(Ps, res_f)
                dt = true_t + t
                dR = RM.dot(true_RM.T)
                max_trans_R, E_Max = Max_displacement(dR, R_axis, dt)
                Max_dis_6_4.append(E_Max)
                Max_dis_R_6_4.append(max_trans_R)
                RMS_ls_6_4.append(RMS(dR, dt, res_f))
                if(RMS(dR, dt, res_f) > 1):
                    print ix, rot_ax, rang, rep
                if(RMS(dR, dt, res_f) > E_Max):
                    print rot_ax, rang, rep, rot_angle
                    print RMS(dR, dt, res_f), E_Max
                RMS_rotation_ls_6_4.append(RMS_Rotation(dR))
                RMS_translation_ls_6_4.append(RMS_Translation(dt))
                RMS_trans_R_6_4.append(RMS_Translation_R(dR))
                counter_6_4.append(counter)

                res = '8mm'
                res_f =  8.0
                rad = res_to_rad(res)
                output_file = '%s_%s_%s_rot_%s_deg_%s_trans.txt' % (res, interp, rot_ax, rang, axes_dict[rot_ax])
                #for algo in xrange(8):
                idx = algo+(rep-1)*8
                counter = results[idx,1]
                Ps = results[idx,2:]
                R_axis = Ps[3:]/np.linalg.norm(Ps[3:])
                t , RM = get_params(Ps, res_f)
                dt = true_t + t
                dR = RM.dot(true_RM.T)
                max_trans_R, E_Max = Max_displacement(dR, R_axis, dt)
                Max_dis_8.append(E_Max)
                Max_dis_R_8.append(max_trans_R)
                RMS_ls_8.append(RMS(dR, dt, res_f))
                if(RMS(dR, dt, res_f) > 1):
                    print ix, rot_ax, rang, rep
                if(RMS(dR, dt, res_f) > E_Max):
                    print rot_ax, rang, rep, rot_angle
                    print RMS(dR, dt, res_f), E_Max
                RMS_rotation_ls_8.append(RMS_Rotation(dR))
                RMS_translation_ls_8.append(RMS_Translation(dt))
                RMS_trans_R_8.append(RMS_Translation_R(dR))
                counter_8.append(counter)

                res = '10mm'
                res_f =  10.0
                rad = res_to_rad(res)
                output_file = '%s_%s_%s_rot_%s_deg_%s_trans.txt' % (res, interp, rot_ax, rang, axes_dict[rot_ax])   
                #for algo in xrange(8):
                idx = algo+(rep-1)*8        
                counter = results[idx,1]
                Ps = results[idx,2:]
                R_axis = Ps[3:]/np.linalg.norm(Ps[3:])
                t , RM = get_params(Ps, res_f)
                dt = true_t + t
                dR = RM.dot(true_RM.T)
                max_trans_R, E_Max = Max_displacement(dR, R_axis, dt)
                Max_dis_10.append(E_Max)
                Max_dis_R_10.append(max_trans_R)
                RMS_ls_10.append(RMS(dR, dt, res_f))
                if(RMS(dR, dt, res_f) > 1):
                    print ix, rot_ax, rang, rep
                if(RMS(dR, dt, res_f) > E_Max):
                    print rot_ax, rang, rep, rot_angle
                    print RMS(dR, dt, res_f), E_Max
                RMS_rotation_ls_10.append(RMS_Rotation(dR))
                RMS_translation_ls_10.append(RMS_Translation(dt))
                RMS_trans_R_10.append(RMS_Translation_R(dR))
                counter_10.append(counter)
                
    return [true_params,
            RMS_ls_6_4,RMS_ls_8,RMS_ls_10,
            RMS_rotation_ls_6_4,RMS_rotation_ls_8,RMS_rotation_ls_10, 
            RMS_translation_ls_6_4,RMS_translation_ls_8,RMS_translation_ls_10,
            Max_dis_6_4,Max_dis_8,Max_dis_10,
            Max_dis_R_6_4,Max_dis_R_8,Max_dis_R_10,
            RMS_trans_R_6_4, RMS_trans_R_8, RMS_trans_R_10,
            counter_6_4, counter_8, counter_10]