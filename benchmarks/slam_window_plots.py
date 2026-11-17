"""Validated covariance ellipse geometry for Stage 4C scientific figures."""

import numpy as np


CONDITIONAL_LABEL = 'Conditional EKF covariance; spline coefficients treated as fixed.'


def covariance_ellipse(covariance, threshold=5.991):
    """Return full width, full height and major-axis angle in degrees.

    Validate the actual marginal; do not clip eigenvalues, symmetrize, or repair
    it. PSD validation of the full active covariance is a separate precondition.
    """
    P = np.asarray(covariance, dtype=np.float64)
    if (P.shape != (2, 2) or not np.isfinite(P).all() or np.max(np.abs(P-P.T)) > 1e-10
            or not np.isfinite(threshold) or threshold <= 0):
        raise ValueError('ellipse requires a finite symmetric 2x2 covariance and positive threshold')
    values, vectors = np.linalg.eigh(P)
    if values[0] < 0:
        raise ValueError('negative marginal covariance eigenvalue; no repair permitted')
    direction = vectors[:, -1]
    return (float(2*np.sqrt(threshold*values[-1])), float(2*np.sqrt(threshold*values[0])),
            float(np.rad2deg(np.arctan2(direction[1], direction[0]))))


def plot_experiment(output, label, arrays, full_truth, full_times):
    """Save five PNG/PDF figures in the anchored frame, validating marginals."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    colors = {'baseline':'#0072B2','optimized':'#D55E00'}
    names = {'baseline':'Baseline B (zero disturbance)','optimized':'Input-spline MHE-SLAM'}
    styles = {'baseline':'--','optimized':'-'}
    for estimator in names:
        for x, P, active in zip(arrays[estimator+'_x'], arrays[estimator+'_covariance'], arrays[estimator+'_active']):
            selected = np.r_[np.arange(3),np.flatnonzero(np.repeat(active,2))+3]
            block = P[np.ix_(selected,selected)]
            if (not np.isfinite(x).all() or not np.isfinite(P).all()
                    or np.max(np.abs(block-block.T))>1e-10 or np.linalg.eigvalsh((block+block.T)/2).min() < -1e-10):
                raise ValueError('invalid active covariance history; plotting aborted without repair')

    def ellipse(axis,center,P,color):
        width,height,angle=covariance_ellipse(P)
        axis.add_patch(Ellipse(center,width,height,angle=angle,fill=False,edgecolor=color,lw=.8,alpha=.65))

    paths=[]
    with plt.rc_context({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300}):
        def save(fig,suffix,conditional=False):
            if conditional:
                fig.text(.5,.01,CONDITIONAL_LABEL,ha='center',fontsize=8)
            fig.tight_layout(rect=(0,.04 if conditional else 0,1,1))
            for extension in ('png','pdf'):
                path=output/f'{label}_{suffix}.{extension}'
                fig.savefig(path,bbox_inches='tight')
                paths.append(path.name)
            plt.close(fig)

        title=f"Configuration {label[0].upper()} — {label.split('_',1)[1]} disturbance"
        fig,axis=plt.subplots(figsize=(7,5))
        prefix=full_times<=arrays['times'][0]
        axis.plot(full_truth[prefix,0],full_truth[prefix,1],':',color='0.55',label='Pre-window truth')
        axis.plot(arrays['truth'][:,0],arrays['truth'][:,1],color='black',lw=1.4,label='Window truth')
        for estimator in names:
            pose,P=arrays[estimator+'_x'],arrays[estimator+'_covariance']
            axis.plot(pose[:,0],pose[:,1],styles[estimator],color=colors[estimator],label=names[estimator])
            for index in np.linspace(0,len(pose)-1,6,dtype=int):
                ellipse(axis,pose[index,:2],P[index,:2,:2],colors[estimator])
        axis.scatter(*arrays['truth'][0,:2],marker='s',s=45,color='#009E73',zorder=4,label='$t_0=2.0$ s: initialization ends / window starts')
        axis.set(xlabel='$p_x$ (m)',ylabel='$p_y$ (m)',title=title+'\n95% robot-position marginals',aspect='equal')
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
        save(fig,'trajectory_covariance',True)

        fig,axis=plt.subplots(figsize=(7,6))
        axis.scatter(*arrays['landmark_truth'].T,marker='x',s=45,color='black',label='True landmarks')
        for estimator in names:
            x,P,active,ids=(arrays[estimator+'_'+field][-1] for field in ('x','covariance','active','truth_ids'))
            landmarks=x[3:].reshape(-1,2)
            for slot in np.flatnonzero(active):
                before = bool(np.any(arrays[estimator+'_truth_ids'][0][arrays[estimator+'_active'][0]] == ids[slot]))
                axis.scatter(*landmarks[slot],marker='^' if before else 'o',s=35,
                             facecolors='none',edgecolors=colors[estimator],
                             label=names[estimator] if slot == np.flatnonzero(active)[0] else None)
                indices=np.arange(3+2*slot,5+2*slot)
                ellipse(axis,landmarks[slot],P[np.ix_(indices,indices)],colors[estimator])
                if estimator=='optimized':axis.annotate(str(ids[slot]),landmarks[slot],xytext=(4,4),textcoords='offset points')
        axis.set(xlabel=r'$\ell_x$ (m)',ylabel=r'$\ell_y$ (m)',aspect='equal',
            title=title+'\nWindow-end 95% marginals; triangles: pre-window births; circles: in-window births')
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
        save(fig,'landmark_covariance',True)

        fig,axes=plt.subplots(3,1,figsize=(8,7),sharex=True)
        for estimator in names:
            error=arrays[estimator+'_x'][:,:3]-arrays['truth']
            error[:,2]=np.arctan2(np.sin(error[:,2]),np.cos(error[:,2]))
            diagonal=np.diagonal(arrays[estimator+'_covariance'],axis1=1,axis2=2)[:,:3]
            if np.any(diagonal<0):raise ValueError('negative robot variance; no repair permitted')
            standard=np.sqrt(diagonal)
            for component,axis in enumerate(axes):
                axis.plot(arrays['times'],error[:,component],styles[estimator],color=colors[estimator],label=names[estimator])
                axis.plot(arrays['times'],2*standard[:,component],':',color=colors[estimator],lw=.8)
                axis.plot(arrays['times'],-2*standard[:,component],':',color=colors[estimator],lw=.8)
                axis.set_ylabel(['$p_x$ error (m)','$p_y$ error (m)','Heading error (rad)'][component])
                axis.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        axes[0].set_title(title+'\n'+r'Errors and componentwise $\pm2$ standard deviations')
        axes[-1].set_xlabel('Absolute time (s)')
        save(fig,'errors_uncertainty',True)

        fig,axes=plt.subplots(2,1,figsize=(8,5),sharex=True)
        for component,axis in enumerate(axes):
            axis.plot(arrays['elapsed'],arrays['disturbance_truth'][:,component],color='black',label='Analytic truth')
            axis.axhline(0,ls='--',color=colors['baseline'],label='Baseline B zero disturbance')
            axis.plot(arrays['elapsed'],arrays['optimized_disturbance'][:,component],color=colors['optimized'],label='MHE-SLAM spline')
            axis.set_ylabel('$w_v$ (m/s)' if component==0 else r'$w_\omega$ (rad/s)')
            axis.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        axes[0].set_title(title)
        axes[-1].set_xlabel('Elapsed optimization-window time (s)')
        save(fig,'disturbances')

        fig,axis=plt.subplots(figsize=(7,3.5))
        axis.plot(arrays['optimization_iteration'],arrays['optimization_objective'],color=colors['optimized'])
        axis.set(xlabel='Accepted optimization iteration (0 = zero start)',ylabel='Innovation quadratic $J$',title=title)
        axis.grid(alpha=.2)
        save(fig,'objective_history')
    return paths


def plot_slam_map(output, label, arrays, sensor):
    """Combined world map, with recorded scan evidence and conditional marginals.

    Call plot_experiment first to validate the complete covariance histories.
    The sensor overlay uses truth; no displayed truth enters an estimator.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, Wedge

    times, truth = arrays['times'], arrays['truth']
    # Choose a scheduled scan nearest the window midpoint, including empty scans.
    scheduled = arrays['scan_scheduled'].astype(bool)
    candidates = np.flatnonzero(scheduled)
    index = candidates[np.argmin(np.abs(times[candidates]-(times[0]+times[-1])/2))]
    selected_times = np.linspace(0, len(times)-1, 6, dtype=int)
    colors = {'baseline':'#0072B2', 'optimized':'#E69F00'}
    names = {'baseline':'Baseline B', 'optimized':'Input-spline'}
    fig, axis = plt.subplots(figsize=(12, 9))
    axis.plot(truth[:,0], truth[:,1], color='black', label='True trajectory')
    axis.scatter(*arrays['landmark_truth'].T, marker='x', color='black', label='True landmarks', zorder=4)
    for truth_id, position in enumerate(arrays['landmark_truth']):
        axis.annotate(f'L{truth_id}', position, xytext=(5,6), textcoords='offset points')

    def ellipse(center, P, color):
        width, height, angle = covariance_ellipse(P)
        axis.add_patch(Ellipse(center, width, height, angle=angle, fill=False,
                               edgecolor=color, lw=.9, alpha=.8))

    for estimator, color in colors.items():
        x, P, active, ids = (arrays[estimator+'_'+field] for field in ('x','covariance','active','truth_ids'))
        axis.plot(x[:,0], x[:,1], '--' if estimator == 'baseline' else '-', color=color,
                  label=names[estimator]+' trajectory')
        arrival_ids = set(ids[0,active[0]].tolist())
        for slot in np.flatnonzero(active[-1]):
            indices = np.arange(3+2*slot,5+2*slot)
            position = x[-1,indices]
            axis.scatter(*position, marker='^' if ids[-1,slot] in arrival_ids else 'o',
                         facecolors='none', edgecolors=color, s=45)
            ellipse(position, P[-1][np.ix_(indices,indices)], color)
        axis.scatter([], [], facecolors='none', edgecolors=color, label=names[estimator]+' final map + 95% ellipses')
        for node in selected_times:
            ellipse(x[node,:2],P[node,:2,:2],color)
    axis.scatter(*truth[0,:2], marker='s', color='#009E73', s=65, zorder=6,
                 label=f'Window start t={times[0]:g} s')
    axis.set(xlabel='x (m)', ylabel='y (m)', aspect='equal')
    axis.margins(.13)
    axis.autoscale_view()
    limits = axis.get_xlim(), axis.get_ylim()
    pose = truth[index]
    fov = np.rad2deg(sensor['field_of_view_rad'])
    radius = sensor['range_max']
    if radius is None:
        radius = 2*np.hypot(np.ptp(limits[0]),np.ptp(limits[1]))
    angle = np.rad2deg(pose[2])
    footprint = Wedge(pose[:2], radius, angle-fov/2, angle+fov/2,
                      width=radius-sensor['range_min'], facecolor='.6', edgecolor='.5',
                      alpha=.12, label=f'Sensor footprint at t={times[index]:g} s')
    axis.add_patch(footprint)
    # The previous node is immediately before this scan; only already mapped IDs.
    previous_ids = arrays['baseline_truth_ids'][max(index-1,0),arrays['baseline_active'][max(index-1,0)]]
    if len(previous_ids):
        axis.scatter(*arrays['landmark_truth'][previous_ids].T, marker='o', s=150,
                     facecolors='none', edgecolors='.5', label='Mapped before selected scan (ID reference)')
    observed = arrays['mask'][index]
    obs_ids = arrays['observation_ids'][index,observed]
    measurements = arrays['measurements'][index,observed]
    for row, truth_id in zip(measurements,obs_ids):
        endpoint = pose[:2]+row[0]*np.array([np.cos(pose[2]+row[1]),np.sin(pose[2]+row[1])])
        axis.plot([pose[0],endpoint[0]],[pose[1],endpoint[1]],color='.25',lw=.7,alpha=.45)
    if len(obs_ids):
        axis.scatter(*arrays['landmark_truth'][obs_ids].T, marker='s', s=220,
                     facecolors='none', edgecolors='black', lw=.8, label='Recorded IDs at selected scan')
    axis.scatter(*pose[:2],marker='o',s=22,color='.3',label=f'True sensor pose t={times[index]:g} s')
    axis.arrow(*pose[:2],.5*np.cos(pose[2]),.5*np.sin(pose[2]),
               width=.015,head_width=.14,color='.3',length_includes_head=True)
    axis.set_xlim(limits[0]); axis.set_ylim(limits[1])
    # Reserve the right margin for the legend and zoom so neither hides map IDs.
    maximum = 'unlimited' if sensor['range_max'] is None else f"{sensor['range_max']:g} m"
    axis.set_title(f"{label}: {sensor['scenario']} sensor, range {sensor['range_min']:g} m–{maximum}, FOV {fov:g}°\n"
                   f"Selected scan t={times[index]:g} s: recorded IDs {obs_ids.tolist()}",fontsize=11)
    axis.legend(loc='upper left',bbox_to_anchor=(1.01,1),fontsize=8,framealpha=.95)
    axis.grid(alpha=.2)
    fig.text(.5,.035,'Robot 95% ellipses: t='+', '.join(f'{t:g}' for t in times[selected_times])+f' s; map ellipses: t={times[-1]:g} s.\n'
             'Map triangles: pre-window births; circles: in-window births. Sight lines: recorded noisy range/bearing.\n'+CONDITIONAL_LABEL,
             ha='center',fontsize=8)
    # Layout the map independently of the manually positioned zoom panel.
    fig.subplots_adjust(left=.07,right=.70,bottom=.16,top=.90)
    inset = fig.add_axes([.74,.19,.22,.32])
    inset.plot(truth[:,0],truth[:,1],color='black')
    for estimator,color in colors.items():
        x=arrays[estimator+'_x']
        inset.plot(x[:,0],x[:,1],'--' if estimator=='baseline' else '-',color=color)
    inset.scatter(*truth[0,:2],marker='s',color='#009E73',s=20)
    inset.set(aspect='equal')
    inset.set_title('Trajectory zoom',fontsize=10)
    inset.tick_params(labelsize=7); inset.grid(alpha=.2)
    paths=[]
    for extension in ('png','pdf'):
        path=output/f'{label}_slam_map.{extension}'
        fig.savefig(path,dpi=220,bbox_inches='tight'); paths.append(path.name)
    plt.close(fig)
    return paths
