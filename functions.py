# -*- coding: utf-8 -*-
"""
Created on Thu May  7 08:38:16 2020

@author: Alvaro
"""

import numpy as np
from pandas import read_csv
import pyvista as pv
import vtk


class trajectory():
    '''
    This class represents a trajectory inputted through a csv file.
    It contains all the functions needed to manipulate the vtk and calculate
    doses.
    '''
    def __init__(self, csv):
        '''
        To initialize the class the csv file name must be inputted.
        The csv file will be automatically parsed into self.traj.
        '''
        self.csv = csv
        # dictionary 'P':points 'vel':velocities 'T':waiting times
        self.traj = self.read_traj()

    def read_traj(self):
        '''
        Returns a dictionary containing the trajectory information from the
        csv file.
        '''
        d = {}  # It will contain 'P', 'T' and 'vel'
        df = read_csv(self.csv)
        X = df['X'].values
        Y = df['Y'].values
        Z = df['Z'].values
        T = df['T'].values
        vel = df['vel'].values

        d['P'] = np.array(list(zip(X, Y, Z)))
        d['T'] = T
        d['vel'] = vel

        return d

    def compute_traj(self, step):
        '''
        Returns a ctrajectory with the given step.
        A ctrajectory has the following structure:
            It is a list of trajectory parts.
            Each trajectory part represents the travel from a point to the
            next in the self.traj.
            Each item in a traj_part is a couple (point, time)
            The first point of every traj_part is repeated. The first copy
            holds the waiting time, the second copy holds the time needed
            to reach the next ctraj point.
            The rest of points in the traj_part hold the time to reach the
            following traj_part point.
        The last traj_part will only have a point as there is no travelling,
        only a waiting time.
        '''
        ctraj = []  # List of traj_parts

        # For all the points minus the last one:
        for i in range(len(self.traj['P'])-1):
            traj_part = []

            # waiting time
            wait_time = self.traj['T'][i]
            # First point in a traj_part holds the waiting time
            traj_part.append([self.traj['P'][i], wait_time])

            # distance between one point and the next
            dist = np.linalg.norm(self.traj['P'][i]-self.traj['P'][i+1])
            # time needed to reach next point
            trav_time = dist/self.traj['vel'][i]

            # If the distance is already lower than the step, there us no need
            # to divide the traj_part in more points.
            if dist < step:
                traj_part.append([self.traj['P'][i], trav_time])
                ctraj.append(traj_part)
                continue

            # director vector for the path
            vdic = np.array(self.traj['P'][i+1])-np.array(self.traj['P'][i])
            vdic = vdic/np.linalg.norm(vdic)  # Normalization

            # Regular paths is the number of times the trip between one point
            # to the next has a distance == step.
            # Irregular path is the last distance as a proportion of the step.
            regular_paths = dist//step
            irregular_path = (dist/step)-regular_paths
            # Time needed to cover a step distance.
            regular_time = trav_time/(regular_paths+irregular_path)
            for j in range(int(regular_paths)):
                traj_part.append([self.traj['P'][i]+vdic*j*step, regular_time])

            if irregular_path > 0:
                traj_part.append([self.traj['P'][i]+vdic*(regular_paths)*step,
                                  trav_time - (regular_paths * regular_time)])

            ctraj.append(traj_part)

        # the last point is added at the end,
        # the time is only the waiting time, vel has no use
        ctraj.append([[self.traj['P'][-1], self.traj['T'][-1]]])

        return ctraj

    def flat_ctraj(self, ctraj):
        '''
        Flattens a ctraj obtained with compute_traj, that is, puts all the
        points and times in a single list, removing the repeated initial
        points.
        This function is useful when we dont need to divide the times
        between waiting and moving time or when we just want the geometric
        location of the points through a trajectory.
        '''
        res = []  # A list of couples (point, time)
        for traj_part in ctraj:
            if len(traj_part) == 2:  # step was bigger than dist
                res.append([traj_part[0][0],
                            traj_part[0][1] + traj_part[1][1]])
            if len(traj_part) > 2:  # typical trajectory
                res.append([traj_part[0][0],
                            traj_part[0][1] + traj_part[1][1]])
                for p, t in traj_part[2:]:
                    res.append([p, t])
            if len(traj_part) == 1:  # last point
                res.append(traj_part[0])
        return res

    def make_lines_from_points(self, points):
        '''
        Given an array of points (np.array), make a line set.
        '''
        poly = pv.PolyData()
        poly.points = points  # points must be a np.array
        cells = np.full((len(points)-1, 3), 2, dtype=np.int)
        cells[:, 1] = np.arange(0, len(points)-1, dtype=np.int)
        cells[:, 2] = np.arange(1, len(points), dtype=np.int)
        poly.lines = cells
        return poly

    def calculate_dose(self, dmap, array_name, step, bounds=[0, 0, 0]):
        '''
        Returns a dictionary of  doses.
        bounds is the size of the box, if zero: it is just a point. Otherwise
        has the form [x, y ,z] for a box
                            -0.5x, +0.5x, -0.5y, +0.5y, -0.5z, +0.5z
        '''
        dose = {'instant dose rate': [],  # dose at the point
                'wait dose':    [],  # dose at the point * waiting time
                'move dose':    [],  # dose during movement * time of movement
                'int dose':     [0.]}  # dose for every point of ctraj

        ctraj = self.compute_traj(step)
        bounds = [-bounds[0]*0.5,
                  +bounds[0]*0.5,
                  -bounds[1]*0.5,
                  +bounds[1]*0.5,
                  -bounds[2]*0.5,
                  +bounds[2]*0.5]

        for traj_part in ctraj:
            p, t = traj_part[0]  # first point corresponds to waiting time
            p = np.array([p[0], p[0], p[1], p[1], p[2], p[2]])
            bnds = p + bounds
            cellIds = vtk.vtkIdList()
            dmap.locator.FindCellsWithinBounds(bnds, cellIds)
            cellIds = [cellIds.GetId(i)
                       for i in range(cellIds.GetNumberOfIds())]
            instant_dose = 0.
            for i in cellIds:
                instant_dose += dmap.mesh[array_name][i]
            if instant_dose == 0:  # to avoid a division by zero
                dose['instant dose rate'].append(instant_dose)
            else:
                dose['instant dose rate'].append(instant_dose/len(cellIds))
            dose['wait dose'].append(dose['instant dose rate'][-1]*t)
            dose['int dose'].append(dose['int dose'][-1]+dose['wait dose'][-1])
            if len(traj_part) == 1:  # final point
                dose['move dose'].append(0)
                # to remove the initial 0.
                dose['int dose'] = dose['int dose'][1:]
                return dose

            move_dose = 0.  # to accumulate all the movement doses of traj_part
            for p, t in traj_part[1:]:
                p = np.array([p[0], p[0], p[1], p[1], p[2], p[2]])
                bnds = p + bounds
                cellIds = vtk.vtkIdList()
                dmap.locator.FindCellsWithinBounds(bnds, cellIds)
                cellIds = [cellIds.GetId(i)
                           for i in range(cellIds.GetNumberOfIds())]
                mdose = 0.  # movement dose of this iteration
                for i in cellIds:
                    mdose += dmap.mesh[array_name][i]
                if mdose == 0:  # to avoid a division by zero
                    mdose = mdose
                else:
                    mdose = mdose/len(cellIds)*t
                dose['int dose'].append(dose['int dose'][-1] + mdose)
                move_dose += mdose
            dose['move dose'].append(move_dose)

    def write_doses(self, dose, result_filename):
        '''
        Writes into a csv file the dose information.
        '''
        df = read_csv(self.csv)
        df['instant dose rate'] = dose['instant dose rate']
        df['wait dose'] = dose['wait dose']
        df['move dose'] = dose['move dose']
        # Iterate over each row to obtain the integrated dose.
        idose = [0.]  # integrated dose
        for i in range(len(dose['instant dose rate'])):
            idose.append(idose[-1]
                         + dose['wait dose'][i]
                         + dose['move dose'][i])
        idose.pop(0)  # to remove the initial 0. value
        df['integral dose'] = idose

        df.to_csv(result_filename, index=False, float_format='%.4E')


class vtk_file():
    '''
    This class represents a vtk file.
    It is processed at the initialization so all its point arrays become cell
    arrays.
    '''
    def __init__(self, filename):
        '''
        It loads the vtk file.
        '''
        # PolyData object of Pyvista
        # All its point arrays are converted to cell arrays.
        self.mesh = pv.read(filename).point_data_to_cell_data()
        # A locator is built to get the cells inside certain bounds in
        # other functions of the trajectory class.
        self.locator = vtk.vtkCellTreeLocator()
        self.locator.SetDataSet(self.mesh)
        self.locator.BuildLocator()


if __name__ == "__main__":
    dose_map = vtk_file('radmap.vtk')
    t = trajectory('traj.csv')
    integral_dose = t.calculate_dose(dose_map,
                                     array_name='scalars',
                                     step=1,
                                     bounds=[2, 3, 1])
