import open3d as o3d 
import copy
import numpy as np 
import math


# read demo point cloud provided by Open3D
pcd_point_cloud = o3d.data.PCDPointCloud()
pcd = o3d.io.read_point_cloud("C:/Users/vegesna/open3d_data/extract/PCDPointCloud/fragment.pcd")
print("Reading Point Cloud Data Successful....... \n")
#print(" The number of Points in Point Cloud Data File are :", pcd)
#print(np.asarray(pcd.points), " pcd point cloud points")

pcd= np.asarray(pcd.points)   #Converting the point cloud file as an array


#Defining RANSAC function 
def RANSAC_funtion(pcd, size, Treshold, Iterations): #pcd is point cloud data, size is number of randomly selected points, 
                                                      #Treshold is the distance between points in point cloud

    for j in range(Iterations):    #For loop to run the code for given number of iterations
        Rand_points= np.random.randint(0, len(pcd)-1, size) #randomly slecting 3 points from 0 to pcd length 
        

        point_1= pcd[Rand_points[0]]    #defining the randomly slected points index from pcd point cloud data
        point_2= pcd[Rand_points[1]]
        point_3= pcd[Rand_points[2]]
        

        x1,y1,z1= point_1[0],point_1[1],point_1[2]  #Defining x,y,z using the randomly slected points 
        x2,y2,z2= point_2[0],point_2[1],point_2[2]
        x3,y3,z3= point_3[0],point_3[1],point_3[2]
        
        a= (((y2-y1)*(z3-z1)) - ((z2-z1)*(y3-y1)))  #Defining formula for Plane coefficcients 
        b= (((z2-z1)*(x3-x1)) - ((x2-x1)*(z3-z1)))
        c= (((x2-x1)*(y3-y1)) - ((y2-y1)*(x3-x1)))
        d= - ((a*x1) + (b*y1) + (c*z1))
        

        #calculating the distance between each point in point cloud data and the plane generated from point cloud data
        distance= np.abs(((a*pcd[:,0]) + (b*pcd[:,1]) + (c*pcd[:,2]) + d)/(math.sqrt(np.abs((a*a) + (b*b) + (c*c)))))
        distance= np.round(distance, 2)
        
        plane_parameters= [a,b,c,d]
        
        index=([])
        inliers=([])
        most_inliers=([])
        
        #Defining a loop in a way that whenever the treshold distance is less than the distance between each point and plane
        i=0 
        for i in range(len(distance)):
            if distance[i] <= Treshold :
                index.append(i)
        
        inliers= pcd[index]
        
        #If loop to select the plane with more number of points within its threshold distance
        if len(inliers)> len(most_inliers):

            plane_coefficients= plane_parameters
            segmentation_inliers= inliers
            segmentation_inliers_index= index
    return plane_coefficients, segmentation_inliers, segmentation_inliers_index

#Defining the parameters for the RANSAC functopm

no_of_random_points= 3
Treshold_distance= 0.1
No_of_Iterations= 1000

#Calling RANSAC Function
plane_coefficients, segmentation_inliers, segmentation_inliers_index= RANSAC_funtion(pcd, no_of_random_points, Treshold_distance, No_of_Iterations)


#print(len(segmentation_inliers_index))

#Sometimes while running the function once the best plane is not selected because the random function may not select the points for best plane 
# So defining a while loop case with condition of if number of points in best plane are less than 75% of point cloud data points the RANSAC 
#function is again Ran to find the best plane fit

while len(segmentation_inliers_index)< 75000:
    print("Computing again since best 3 points in a plane are not selected during random point selection..........")
    plane_coefficients, segmentation_inliers, segmentation_inliers_index= RANSAC_funtion(pcd, no_of_random_points, Treshold_distance, No_of_Iterations)
    print("The number of Inliers are :::",len(segmentation_inliers_index)) 
    
print("\nThe plane Equation is ::: %s x + %s y + %s z + %s "
         %(plane_coefficients[0],plane_coefficients[1],plane_coefficients[2],plane_coefficients[3]))





segmentation_inliers=np.concatenate(segmentation_inliers, axis=0)


pcd_point_cloud = o3d.data.PCDPointCloud()
pcd = o3d.io.read_point_cloud("C:/Users/vegesna/open3d_data/extract/PCDPointCloud/fragment.pcd")

inliers_array= pcd.select_by_index(np.asarray(segmentation_inliers_index))
outlier_cloud = pcd.select_by_index(np.asarray(segmentation_inliers_index), invert=True)

#coloring the best plane to red 
inliers_array.paint_uniform_color([1.0, 0, 0])

# Visualising the point cloud data 

o3d.visualization.draw_geometries([inliers_array, outlier_cloud],
                                  zoom=0.8,
                                front=[-0.4999, -0.1659, -0.8499],
                               lookat=[2.1813, 2.0619, 2.0999],up=[0.1204, -0.9852, 0.1215])


