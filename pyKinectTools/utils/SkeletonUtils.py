#from pylab import *
import numpy as np
import cv2
import cv2.cv as cv
import time



# ----------------------------------------------------------------

from skimage.draw import circle, line
def display_MSR_skeletons(img, skel, color=(200,0,0), skel_type='MSR'):
    '''
    skel_type : 'MSR' or 'Low' ##or 'Upperbody'
    '''
    if skel_type == 'MSR':
        joints = range(20)
        connections = [
                        [3, 2],[2,1],[1,0], #Head to torso
                        [2, 4],[4,5],[5,6],[6,7], # Left arm
                        [2, 8],[8,9],[9,10],[10,11], # Right arm
                        [0,12],[12,13],[13,14],[14,15], #Left foot
                        [0,16],[16,17],[17,18],[18,19]
                        ]
    elif skel_type == 'Low':
        joints = [0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 13, 15, 17, 19]
        connections = [
                        [3, 2],[2,1],[1,0], #Head to torso
                        [2, 4],[4,5],[5,7], # Left arm
                        [2, 8],[8,9],[9,11], # Right arm
                        [0,13],[13,15], #Left foot
                        [0,17],[17,19]
                        ]
    elif skel_type == 'Upperbody':
        joints = [0, 1, 2, 3, 4, 5, 7, 8, 9, 11]
        connections = [
                        [3, 2],[2,1],[1,0], #Head to torso
                        [2, 4],[4,5],[5,7], # Left arm
                        [2, 8],[8,9],[9,11], # Right arm
                        ]

    for i in joints:
        j = skel[i]
        # Remove zero nodes
        if not (j[0] == 0 and j[1] == 0):
            cv2.circle(img, (j[0], j[1]), 5, color)

    # Make head a bigger node
    cv2.circle(img, (skel[3,0], skel[3,1]), 15, color)

    for c in connections:
        # Remove zero nodes
        if not ( (skel[c[0],0]==0 and skel[c[0],1]==0) or (skel[c[1],0]==0 and skel[c[1],1]==0)):
            cv2.line(img, (skel[c[0],0], skel[c[0],1]), (skel[c[1],0], skel[c[1],1]), color, 2)

    return img


''' ----- Old data ---- '''


#ENUMS
colors = 'kkkkkkrgkbkkrgkbkcmkycmkykkkkkk'
S_HEAD      =1
S_NECK      =2
S_TORSO     =3
S_L_SHOULDER    =6
S_L_ELBOW   =7
S_L_HAND        =9
S_R_SHOULDER    =12
S_R_ELBOW   =13
S_R_HAND        =15
S_L_HIP     =17
S_L_KNEE        =18
S_L_FOOT        =20
S_R_HIP     =21
S_R_KNEE        =22
S_R_FOOT        =24
# SKELETON = [S_HEAD, S_NECK, S_TORSO, S_L_SHOULDER, S_L_ELBOW, S_L_HAND, S_R_SHOULDER, S_R_ELBOW, S_R_HAND,  S_L_HIP, S_L_KNEE, S_L_FOOT, S_R_HIP, S_R_KNEE, S_R_FOOT]
# SKELETON = [S_HEAD, S_L_SHOULDER, S_L_ELBOW, S_L_HAND, S_R_SHOULDER, S_R_ELBOW, S_R_HAND, S_L_KNEE, S_L_FOOT, S_R_KNEE, S_R_FOOT] # removed neck, torso, hip
SKELETON = [S_HEAD, S_L_SHOULDER, S_L_ELBOW, S_L_HAND, S_R_SHOULDER, S_R_ELBOW, S_R_HAND, S_L_FOOT, S_R_FOOT] # removed neck, torso, hip, knee
# SKELETON = [S_HEAD, S_L_ELBOW, S_L_HAND, S_R_ELBOW, S_R_HAND, S_L_FOOT, S_R_FOOT] # removed neck, torso, shoulders, hip, knee

#PARENTS
P_HEAD          = S_NECK
P_NECK          = S_TORSO
P_TORSO         = S_TORSO
P_L_SHOULDER    = S_NECK
P_L_ELBOW       = S_L_SHOULDER
# P_L_HAND      = S_L_SHOULDER#S_L_ELBOW # displacement from shoulder is better metric! (6% gain)
P_L_HAND        = S_L_ELBOW # displacement from shoulder is better metric! (6% gain)
P_R_SHOULDER    = S_NECK
P_R_ELBOW       = S_R_SHOULDER
# P_R_HAND      = S_R_SHOULDER#S_R_ELBOW
P_R_HAND        = S_R_ELBOW # displacement from shoulder is better metric! (6% gain)
P_L_HIP         = S_TORSO
P_L_KNEE        = S_L_HIP
P_L_FOOT        = S_L_KNEE
P_R_HIP         = S_TORSO
P_R_KNEE        = S_R_HIP
P_R_FOOT        = S_R_KNEE
# PARENTS = [P_HEAD, P_NECK, P_TORSO, P_L_SHOULDER, P_L_ELBOW, P_L_HAND, P_R_SHOULDER, P_R_ELBOW, P_R_HAND,  P_L_HIP, P_L_KNEE, P_L_FOOT, P_R_HIP, P_R_KNEE, P_R_FOOT]
# PARENTS = [P_HEAD, P_L_SHOULDER, P_L_ELBOW, P_L_HAND, P_R_SHOULDER, P_R_ELBOW, P_R_HAND, P_L_KNEE, P_L_FOOT, P_R_KNEE, P_R_FOOT] # removed neck, torso
PARENTS = [P_HEAD, P_L_SHOULDER, P_L_ELBOW, P_L_HAND, P_R_SHOULDER, P_R_ELBOW, P_R_HAND, P_L_FOOT, P_R_FOOT] # removed neck, torso, hip
# PARENTS = [P_HEAD, P_L_ELBOW, P_L_HAND, P_R_ELBOW, P_R_HAND, P_L_FOOT, P_R_FOOT] # removed neck, torso, shoulders,  hip
# PARENTS = [P_TORSO, P_TORSO, P_TORSO, P_TORSO, P_TORSO, P_TORSO, P_TORSO, P_TORSO, P_TORSO, P_TORSO, P_TORSO] # removed neck, torso




def readUserData(file):
    raw = open(file)
    raw = raw.read()
    sp = raw.split()
    users = {}

    while ('User' in sp):
        index = sp.index('User')
        users[int(sp[index+1])] = {}
        x = float(sp[index+2])
        y = float(sp[index+3])
        z = float(sp[index+4])
        users[int(sp[index+1])]['World'] = [x, y, z]
        imgCoords = world2depth([x, y, z])
        users[int(sp[index+1])]['Img'] = imgCoords
        sp.pop(index)
        # print users
    return users


# set normOrn to 0!
def readSkeletonData(file, normalizeOrientation=0):
    
    raw = open(file)
    raw = raw.read()
    sp = raw.split()
    
    skeletons = {}    

    while 'Skeleton' in sp:    
        baseInd = sp.index('Skeleton')
        skelInd = int(sp[baseInd+1])
        if skeletons.has_key(skelInd) == 0:
            skeletons[skelInd] = {}
            skeletons[skelInd]['rawJoints'] = []
        
        for j in range(25):
            for k in range(3):
                skeletons[skelInd]['rawJoints'].append(float(sp[baseInd+j*3+k+2]))
            err = np.array(skeletons[skelInd]['rawJoints'][-3:])
            if np.sum(err) == 0:
                inds = [len(skeletons[skelInd]['rawJoints'])-3]
                inds.append(inds[0]+3)
                new_data = filter(skeletons[skelInd]['rawJoints'], err, inds)
                skeletons[skelInd]['rawJoints'][inds[0]:inds[1]] = new_data
        sp.pop(baseInd)
    
    for i in skeletons.keys():
        joints = np.array(skeletons[i]['rawJoints'], dtype=float)
        joints = joints.reshape(-1, 25, 3)
        jointsN = joints.copy()
        jointsRel = joints.copy()
        
        import pdb
        # pdb.set_trace()

        # Get centered positions
        for j in range(len(joints)):
            jointsN[j] = joints[j] - joints[j, S_TORSO]
            # pdb.set_trace()
            # Normalize orientation:
            if normalizeOrientation:
                # nVecs = np.array([jointsN[j, S_TORSO],  jointsN[j, S_L_SHOULDER], jointsN[j, S_R_SHOULDER]])
                nVecs = np.array([jointsN[j, S_TORSO],  jointsN[j, S_L_SHOULDER], jointsN[j, S_R_SHOULDER], jointsN[j, S_NECK], jointsN[j, S_L_HIP],jointsN[j, S_R_HIP]])
                # jointsN[j, S_NECK],
                nVecs -= nVecs.T.mean(1)
                _,_,v = np.linalg.svd(nVecs)
                # pdb.set_trace()
                a = np.array(v[:,2]) # Get normal
                if a[2] > 0:
                    a = -a
                # print a
                b = np.array([0.0,0.0,-1.0])
                
                # Get axis angle:
                angle = -np.arccos(np.dot(a,b)) # = cos(theta)
                axis = np.cross(a,b)
                axis /= np.linalg.norm(axis)
                # print angle*180.0/np.pi, axis                
                # Apply orientation normalization to all joints
                # print jointsN[j,:]
                for k in range(len(SKELETON)):
                    # pdb.set_trace()
                    v = jointsN[j,SKELETON[k]]
                    # pdb.set_trace()
                    # v /= np.sqrt(np.sum(v**2))
                    vOut = v * np.cos(angle) + np.cross(axis, v)*np.sin(angle) + axis*(np.dot(axis, v))*(1-np.cos(angle))
                    jointsN[j,SKELETON[k]] = vOut
                    # print vOut
                    # pdb.set_trace()

                nVecs = np.array([jointsN[j, S_TORSO], jointsN[j, S_L_SHOULDER], jointsN[j, S_R_SHOULDER]])
                nVecs -= nVecs.T.mean(1)
                _,_,v = np.linalg.svd(nVecs)
                # print v
                # pdb.set_trace()

            

        # Get relative positions
        for j in range(len(SKELETON)):
            # jointsRel[:,SKELETON[j]] = joints[:,SKELETON[j]] - joints[:,PARENTS[j]]
            jointsRel[:,SKELETON[j]] = jointsN[:,SKELETON[j]] - jointsN[:,PARENTS[j]]
            tmp = np.mean(np.sqrt(np.sum(jointsRel[:,SKELETON[0]]**2, axis=1)))
            # if not np.any(np.isnan(tmp)) and not np.any(tmp==0):
            jointsRel[:,SKELETON[j]] /= np.mean(np.sqrt(np.sum(jointsRel[:,SKELETON[0]]**2, axis=1))) # normalize

        skeletons[i]['jointsPos'] = joints
        skeletons[i]['jointsCentered'] = jointsN
        skeletons[i]['jointsRel'] = jointsRel

    return skeletons



### Filter ###
def filter(data, check, inds):
#    print check
#    print inds
    if np.sum(check) == 0:
        new_inds = [inds[0]-78,inds[1]-78]    
        if new_inds[0] < 0 or new_inds[1] < 0: #if the beginning of the file
            return np.array([0.,0.,0.])
#            print "<0"
        else:
            check = np.array(data[new_inds[0] : new_inds[1]])
            check = filter(data, check, new_inds)
    return check
        
#    skels = readSkeletonData(file)

# ----------------------------------------------------------------

def displaySkeleton(skeleton):    
    
    joints = skeleton['jointsPos']
    jointsN = skeleton['jointsCentered']
    jointsRel = skeleton['jointsRel']


    # Get relative positions
    for i in range(len(SKELETON)):
        jointsRel[:,SKELETON[i]] = joints[:,SKELETON[i]] - joints[:,PARENTS[i]]    

    axes(axisbg = [.9,.9,.9])
#    axes(axisbg = [1,1,1])
    for i in range(0, len(joints), 2):
        b = 1100 # Axis size                
        # Center about the torso
        jointsN[i] = joints[i] - joints[i, 3]    
        
        # Chart over time
        if (0):
            colorSet = ['r','g','b', 'c', 'm', 'y', 'k', 'r','g','b', 'c', 'm', 'y', 'k', 'r','g','b', 'c', 'm', 'y', 'k','r','g','b', 'c', 'm', 'y', 'k']
            axis([0, 200, -300, 300])                
            for label in SKELETON:
                plot(i, jointsRel[i,label,0], 'o', color = colorSet[label])

        # Display relative positions
        if (0):
            cla()
            b = 500
            axis([-b, b, -b, b])             
            # Viz head as circle
            plot(jointsRel[i, 1, 0], jointsRel[i,1,1], 'og', markersize=40, markerfacecolor='w')     
            # Viz joints as lines
            for label in SKELETON:
                plot([0, jointsRel[i,label,0]], [0, jointsRel[i,label, 1]])

        # Display absolute positions
        if (0):
            cla()
            b = 1100  
            axis([-b, b, -b, b])                
            # Viz joints as lines
            for label in range(len(SKELETON)):
                plot([joints[i,SKELETON[label],0], joints[i,PARENTS[label],0]], [joints[i,SKELETON[label],1], joints[i,PARENTS[label], 1]])
            # Connect hips
            plot([joints[i, S_L_HIP, 0], joints[i, S_R_HIP, 0]], [joints[i, S_L_HIP, 1], joints[i, S_R_HIP, 1]]) 
            # Viz head as circle
            plot(joints[i, 1, 0], joints[i,1,1], 'og', markersize=40, markerfacecolor='w')  
            # Connect shoulder to torso
            plot([joints[i, S_TORSO, 0], joints[i, S_L_SHOULDER, 0]], [joints[i, S_TORSO, 1], joints[i, S_L_SHOULDER, 1]]) #diag
            plot([joints[i, S_TORSO, 0], joints[i, S_R_SHOULDER, 0]], [joints[i, S_TORSO, 1], joints[i, S_R_SHOULDER, 1]]) #diag  

        # Display centered, absolute positions
        if (1):
            cla()
            b = 1100
            axis([-b, b, -b, b])
            # Viz joints as lines
            for label in range(len(SKELETON)):
                plot([jointsN[i,SKELETON[label],0], jointsN[i,PARENTS[label],0]], [jointsN[i,SKELETON[label],1], jointsN[i,PARENTS[label], 1]])
            # Connect hips
            plot([jointsN[i, S_L_HIP, 0], jointsN[i, S_R_HIP, 0]], [jointsN[i, S_L_HIP, 1], jointsN[i, S_R_HIP, 1]])
            # Viz head as circle
            plot(jointsN[i, 1, 0], jointsN[i,1,1], 'og', markersize=40, markerfacecolor='w')     
            # Connect shoulder to torso
            plot([jointsN[i, S_TORSO, 0], jointsN[i, S_L_SHOULDER, 0]], [jointsN[i, S_TORSO, 1], jointsN[i, S_L_SHOULDER, 1]]) #diag
            plot([jointsN[i, S_TORSO, 0], jointsN[i, S_R_SHOULDER, 0]], [jointsN[i, S_TORSO, 1], jointsN[i, S_R_SHOULDER, 1]]) #diag  

        draw()
        time.sleep(.01)

    # Flash screen to show new drawing
    axes(axisbg = [0,.25,0])
    draw()
    time.sleep(.5)

def getAllSkeletons(folder, normalizeOrientation=0, count=55):
    skelSets = {}
    for i in range(count):
        file = folder + '/jointLog' + str(i) + '.txt'
        skelSets[i] = readSkeletonData(file, normalizeOrientation)
    return skelSets


def displaySkeleton_CV(img, skels):
    for i in skels:
        try:
            # joints = skels[i]['jointsPos']
            joints = skels[i]['jointsPos']
            # print i
            # pdb.set_trace()
            if np.any(joints[0, :, 2]) != 0:
                for label in range(len(SKELETON)):
                        pts1 = joints[0, SKELETON[label]]
                        pts1 = world2depth(list(pts1))
                        pts2 = joints[0, PARENTS[label]]
                        pts2 = world2depth(list(pts2))
                        if pts1[0] != 0 and pts2[0] != 0:
                            cv.Line(img, (640-pts1[0], 480-pts1[1]), (640-pts2[0], 480-pts2[1]), [100], 5)
                # Connect hips
                pts1 = joints[0, S_L_HIP]
                pts1 = world2depth(list(pts1))
                pts2 = joints[0, S_R_HIP]
                pts2 = world2depth(list(pts2))
                if pts1[0] != 0 and pts2[0] != 0:
                    cv.Line(img, (640-pts1[0], 480-pts1[1]), (640-pts2[0], 480-pts2[1]), [100], 5)
                # Viz head as circle
                pts1 = joints[0, 1]
                pts1 = world2depth(list(pts1))
                if pts1[0] != 0:
                    cv.Circle(img,(640-pts1[0],480-pts1[1]), 30, [100], 3)
                # Connect shoulder to torso
                pts1 = joints[0, S_TORSO]
                pts1 = world2depth(list(pts1))
                pts2 = joints[0, S_L_SHOULDER]
                pts2 = world2depth(list(pts2))
                if pts1[0] != 0 and pts2[0] != 0:
                    cv.Line(img, (640-pts1[0], 480-pts1[1]), (640-pts2[0], 480-pts2[1]), [100], 5)
                pts1 = joints[0, S_TORSO]
                pts1 = world2depth(list(pts1))
                pts2 = joints[0, S_R_SHOULDER]
                pts2 = world2depth(list(pts2))
                if pts1[0] != 0 and pts2[0] != 0:
                    cv.Line(img, (640-pts1[0], 480-pts1[1]), (640-pts2[0], 480-pts2[1]), [100], 5)

                # plot([joints[i, S_TORSO, 0], joints[i, S_L_SHOULDER, 0]], [joints[i, S_TORSO, 1], joints[i, S_L_SHOULDER, 1]]) #diag
                # plot([joints[i, S_TORSO, 0], joints[i, S_R_SHOULDER, 0]], [joints[i, S_TORSO, 1], joints[i, S_R_SHOULDER, 1]]) #diag           
            # else:
            #     print joints[0]

        except:
            print "error"
            continue
    return img





# jointsN = skelSetsTrain[0][0]['jointsCentered']


def displaySkeleton_3D(img, skels):
    from mpl_toolkits.mplot3d import Axes3D
    fig = figure(7)
    ax = fig.add_subplot(111, projection='3d')
    tmp = [x for x in SKELETON]

    
    for i in range(jointsN.shape[0]):
        ax.cla()
        ax.scatter(jointsN[i,tmp,0], jointsN[i,tmp,1], jointsN[i,tmp,2], zdir='y')

        for label in range(len(SKELETON)):
            ax.plot([jointsN[i,SKELETON[label],0], jointsN[i,PARENTS[label],0]], [jointsN[i,SKELETON[label],1], jointsN[i,PARENTS[label], 1]], [jointsN[i,SKELETON[label],2], jointsN[i,PARENTS[label], 2]], zdir='y')
            # Connect hips
            ax.plot([jointsN[i, S_L_HIP, 0], jointsN[i, S_R_HIP, 0]], [jointsN[i, S_L_HIP, 1], jointsN[i, S_R_HIP, 1]], [jointsN[i, S_L_HIP, 2], jointsN[i, S_R_HIP, 2]], zdir='y')
            # Viz head as circle
            ax.plot([jointsN[i, 1, 0]], [jointsN[i,1,1]], [jointsN[i,1,2]], zdir='y')     
            # Connect shoulder to torso
            ax.plot([jointsN[i, S_TORSO, 0], jointsN[i, S_L_SHOULDER, 0]], [jointsN[i, S_TORSO, 1], jointsN[i, S_L_SHOULDER, 1]], [jointsN[i, S_TORSO, 2], jointsN[i, S_L_SHOULDER, 2]], zdir='y') #diag
            ax.plot([jointsN[i, S_TORSO, 0], jointsN[i, S_R_SHOULDER, 0]], [jointsN[i, S_TORSO, 1], jointsN[i, S_R_SHOULDER, 1]], [jointsN[i, S_TORSO, 2], jointsN[i, S_R_SHOULDER, 2]], zdir='y') #diag  

        # ax.axis([-500, 500, -500, 500])
        # ax.set_aspect('equal', 'box')
        # ax.set_aspect('equal')
        #Set artificual limits
        MAX = 500
        for direction in (-1, 1):
            for point in np.diag(direction * MAX * np.array([1,1,1])):
                ax.plot([point[0]], [point[1]], [point[2]], 'w')

        draw()

def plotSkelAxes():
    labels = [x for x in SKELETON]
    # colors = 'rgbykmrgbykmrgbykmrgbykmrgbykmrgbykmrgbykmrgbykm'
    colors = 'kkkkkkrgkbkkrgkbkcmkycmkykkkkkk'
    #Show projections
    figure(10)
    subplot(1,3,1)
    for i in tmp: #in z direction
        print i
        plot(jointsN[:,i, 0], jointsN[:,i,1], label=str(i), color=colors[i])
        title('Front')
        ylabel('x')
        ylabel('y')    
    subplot(1,3,2)
    for i in tmp: #in x direction
        plot(jointsN[:,i, 1], jointsN[:,i,2], label=str(i), color=colors[i])
        title('Side')
        ylabel('y')
        ylabel('z')    
    subplot(1,3,3)
    for i in tmp: #in x direction
        plot(jointsN[:,i, 0], jointsN[:,i,2], label=str(i), color=colors[i])
        title('Top')
        ylabel('x')
        ylabel('z')


