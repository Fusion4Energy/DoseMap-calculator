# -*- coding: utf-8 -*-
"""
########################################################################################################
# Copyright 2019 F4E | European Joint Undertaking for ITER and the Development                         #
# of Fusion Energy (‘Fusion for Energy’). Licensed under the EUPL, Version 1.2                         #
# or - as soon they will be approved by the European Commission - subsequent versions                  #
# of the EUPL (the “Licence”). You may not use this work except in compliance                          #
# with the Licence. You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl.html       #
# Unless required by applicable law or agreed to in writing, software distributed                      #
# under the Licence is distributed on an “AS IS” basis, WITHOUT WARRANTIES                             #
# OR CONDITIONS OF ANY KIND, either express or implied. See the Licence permissions                    #
# and limitations under the Licence.                                                                   #
########################################################################################################
@author: Alvaro Cubi
"""
from functions import trajectory, vtk_file

import numpy as np
import os
import pyvista as pv
import sys

from PyQt5 import Qt
from PyQt5.QtWidgets import (QFileDialog, QComboBox, QLabel, QMessageBox,
                             QDialog, QTableWidget, QTableWidgetItem,
                             QApplication)


class MainWindow(Qt.QMainWindow):
    '''
    This class contains all the GUI functions.
    It is the Main Window that when initialized will operate the whole
    program.

    The attributes that are not focused on the GUI but on the calculations
    are the following:
        self.trajectories: dictionary containing all the trajectory classes
        self.vtk_files: dictionary containing all the vtk file classes
        self.bounds
        self.step
    '''
    def __init__(self, parent=None, show=True):
        Qt.QMainWindow.__init__(self, parent)
        self.setWindowTitle('DoseMap trajectory calculator')
        self.statusBar().showMessage(
                'Welcome to DoseMap trajectory calculator.')

        # Attributes not related to the GUI
        self.trajectories = {}  # keys=csv filename  values=trajectory class
        self.vtk_files = {}  # keys=vtk filename  values=vtk_file class
        self.bounds = [5, 5, 5]  # boundaries of the moving prism
        self.step = 1  # Maximum distance between computed points
        self.sargs = dict(interactive=True,  # Log bar for plots
                          title_font_size=20,
                          label_font_size=16,
                          shadow=True,
                          n_labels=11,  # Because n_colors of the plot is 10
                          italic=True,
                          fmt="%.e",
                          font_family="arial")

        # create the frame
        self.frame = Qt.QFrame()
        vlayout = Qt.QVBoxLayout()
        hlayout1 = Qt.QHBoxLayout()  # ComboBox labels
        hlayout2 = Qt.QHBoxLayout()  # ComboBox selections
        vlayout.addLayout(hlayout1)
        vlayout.addLayout(hlayout2)

        # add the pyvista interactor object
        self.plotter = pv.QtInteractor(self.frame)
        vlayout.addWidget(self.plotter.interactor)

        # Populate the window with the vlayout
        self.frame.setLayout(vlayout)
        self.setCentralWidget(self.frame)

        # MENU TO LOAD FILES
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')

        opencsvButton = Qt.QAction('Load a trajectory csv file', self)
        opencsvButton.triggered.connect(self.opencsv)
        fileMenu.addAction(opencsvButton)

        openVTKbutton = Qt.QAction('Load a VTK file', self)
        openVTKbutton.triggered.connect(self.openVTK)
        fileMenu.addAction(openVTKbutton)

        exitButton = Qt.QAction('Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)

        # PLOTTING MENU
        plotMenu = mainMenu.addMenu('Plot')
        # plot selected mesh
        plotmesh_action = Qt.QAction('Plot mesh', self)
        plotmesh_action.triggered.connect(self.plotmesh)
        plotMenu.addAction(plotmesh_action)
        # plot selected trajectory
        plottraj_action = Qt.QAction('Plot trajectory', self)
        plottraj_action.triggered.connect(self.plottraj)
        plotMenu.addAction(plottraj_action)
        # show prism
        showprism_action = Qt.QAction('Show prism', self)
        showprism_action.triggered.connect(self.show_prism)
        plotMenu.addAction(showprism_action)
        # plot sampled trajectory over mesh and data array
        plotsampledtraj_action = Qt.QAction('Plot sampled trajectory', self)
        plotsampledtraj_action.triggered.connect(self.sample_over_traj)
        plotMenu.addAction(plotsampledtraj_action)
        # clear slicing planes
        clearplanes_action = Qt.QAction('Clear slicing planes', self)
        clearplanes_action.triggered.connect(self.clear_planes)
        plotMenu.addAction(clearplanes_action)
        # clear plots
        clearplots_action = Qt.QAction('Clear plots', self)
        clearplots_action.setShortcut('Ctrl+C')
        clearplots_action.triggered.connect(self.clear_plots)
        plotMenu.addAction(clearplots_action)

        # PARAMETERS MENU
        paramMenu = mainMenu.addMenu('Parameters')
        # input XYZ bounds
        xyzbounds_action = Qt.QAction('Input XYZ bounds', self)
        xyzbounds_action.triggered.connect(self.xyzbounds)
        paramMenu.addAction(xyzbounds_action)
        # input step
        set_step_action = Qt.QAction('Set step distance', self)
        set_step_action.triggered.connect(self.set_step)
        paramMenu.addAction(set_step_action)
        # csv traj
        csv_traj_action = Qt.QAction('Display trajectory table', self)
        csv_traj_action.triggered.connect(self.csv_traj)
        paramMenu.addAction(csv_traj_action)

        # CALCULATE MENU
        calculateMenu = mainMenu.addMenu('Calculate')
        # dose calculation
        dose_calculation_action = Qt.QAction('Calculate doses', self)
        dose_calculation_action.triggered.connect(self.dose_calculation)
        calculateMenu.addAction(dose_calculation_action)

        # About us menu
        plotMenu = mainMenu.addMenu('About us')
        # about F4E
        aboutf4e_action = Qt.QAction('Fusion for Energy', self)
        aboutf4e_action.triggered.connect(self.aboutf4e)
        plotMenu.addAction(aboutf4e_action)
        # about rpn
        aboutRPN_action = Qt.QAction('Raul Pampin', self)
        aboutRPN_action.triggered.connect(self.aboutRPN)
        plotMenu.addAction(aboutRPN_action)
        # about mfi
        aboutMFI_action = Qt.QAction('Marco Fabbri', self)
        aboutMFI_action.triggered.connect(self.aboutMFI)
        plotMenu.addAction(aboutMFI_action)
        # about aci
        aboutACI_action = Qt.QAction('Alvaro Cubi', self)
        aboutACI_action.triggered.connect(self.aboutACI)
        plotMenu.addAction(aboutACI_action)
        # about hcn
        aboutHCN_action = Qt.QAction('Haridev Chohan', self)
        aboutHCN_action.triggered.connect(self.aboutHCN)
        plotMenu.addAction(aboutHCN_action)

        # comboBox to select trajectory
        self.cbTrajectory = QComboBox()
        self.cbTrajectory.currentIndexChanged.connect(
                                                self.selectionchangeTrajectory)
        hlayout1.addWidget(QLabel(self, text="Trajectory"))
        hlayout2.addWidget(self.cbTrajectory)

        # comboBox to select vtk file
        self.cbVTK = QComboBox()
        self.cbVTK.currentIndexChanged.connect(self.selectionchangeVTK)
        hlayout1.addWidget(QLabel(self, text="VTK file"))
        hlayout2.addWidget(self.cbVTK)

        # comboBox to select vtk array
        self.cbArray = QComboBox()
        self.cbArray.activated.connect(self.selectionchangeArray)
        hlayout1.addWidget(QLabel(self, text="Data Array"))
        hlayout2.addWidget(self.cbArray)

        if show:
            self.show()

    def opencsv(self):
        '''
        Select a csv and load it as a trajectory class instance.
        '''
        options = QFileDialog.Options()
        csvfile, _ = QFileDialog.getOpenFileName(
                self,
                "Select the trajectory csv file",
                "",
                "CSV files (*.csv);;All Files (*)",
                options=options
                )
        if csvfile is not None and csvfile != '':
            # to extract the filename from the path
            csv = os.path.split(csvfile)[-1]
            if csv in self.trajectories.keys():
                self.statusBar().showMessage(
                        'A csv file with the same name is already loaded.')
                return
            # Using the csv name as key, set as value the csv file path
            self.trajectories[csv] = trajectory(csvfile)
            self.cbTrajectory.addItem(csv)  # Add it to the ComboBox
            self.statusBar().showMessage('CSV trajectory file loaded.')

    def openVTK(self):
        '''
        Select a file that can be read as a PolyData object of Pyvista.
        That includes vtk files containing meshes and info as well as
        stl files that simply show a geometry.
        '''
        options = QFileDialog.Options()
        vtkpath, _ = QFileDialog.getOpenFileName(
                self,
                "Select the VTK file",
                "",
                "All Files (*);;VTK files (*.vtk)",
                options=options
                )
        if vtkpath is not None and vtkpath != '':
            # to extract the filename from the path
            vtk_name = os.path.split(vtkpath)[-1]
            if vtk_name in self.vtk_files.keys():
                self.statusBar().showMessage(
                        'A vtk file with the same name is already loaded.')
                return
            self.statusBar().showMessage('Loading VTK file...')
            self.vtk_files[vtk_name] = vtk_file(vtkpath)
            self.cbVTK.addItem(vtk_name)
            self.statusBar().showMessage('VTK file loaded.')

    def plotmesh(self):
        '''
        Plots a clip plane mesh.
        The mesh is selected from the ComboBox.
        It has an opacity value to see through it.
        I can have no array attached to it.
        If there is an array in the array ComboBox the array data will be
        plotted.
        When plotting data a log scale will be used, therefore only positive
        non-zero values will appear.
        '''
        # Try to select an array for the mesh. If it fails just plot the
        # mesh alone.
        array = self.cbArray.currentText()
        if array is not None and array != '':
            # Apply a treshold to the data of the array so only positive
            # non-zero data will remain. This is needed for the correctnes
            # of the log scale plot. Negative or zero values dont make sense.
            mesh = (self.vtk_files[self.cbVTK.currentText()].mesh.
                    threshold([1e-20, 1e50], scalars=array))
            self.plotter.add_mesh_clip_plane(mesh,
                                             scalars=array,
                                             log_scale=True,
                                             scalar_bar_args=self.sargs,
                                             cmap='jet',
                                             opacity=0.7,
                                             n_colors=10)
        else:
            mesh = self.vtk_files[self.cbVTK.currentText()].mesh
            self.plotter.add_mesh_clip_plane(mesh,
                                             cmap='jet',
                                             color='grey',
                                             opacity=1.)
        self.plotter.show_bounds()
        self.plotter.reset_camera()

    def plottraj(self):
        '''
        Plots the points of a trajectory computed with traj and step.
        '''
        t = self.trajectories[self.cbTrajectory.currentText()]
        ctraj = t.compute_traj(self.step)
        ftraj = t.flat_ctraj(ctraj)
        points = np.array([list(i) for i, z in ftraj])
        lines = t.make_lines_from_points(points)
        self.plotter.add_mesh(lines, color='black', line_width=5.)
        self.plotter.reset_camera()

    def sample_over_traj(self):
        '''
        Similar to plottraj but in this case the plotted lines are sampled
        over a data mesh. The scale is logarithmic.
        '''
        t = self.trajectories[self.cbTrajectory.currentText()]
        ctraj = t.compute_traj(self.step)
        ftraj = t.flat_ctraj(ctraj)
        points = np.array([list(i) for i, z in ftraj])
        lines = t.make_lines_from_points(points)
        sampled = lines.sample(self.vtk_files[self.cbVTK.currentText()].mesh)
        # In case the trajectory starts out of the radmap or has a zero value
        # anywhere the plot bar (which is in log scale) will not be affected.
        if sampled[self.cbArray.currentText()].min() == 0.:
            self.plotter.add_mesh(sampled,
                                  show_scalar_bar=False,
                                  scalars=self.cbArray.currentText(),
                                  log_scale=True,
                                  scalar_bar_args=self.sargs,
                                  cmap='jet',
                                  opacity=1.0,
                                  line_width=5.,
                                  n_colors=10)
        else:
            self.plotter.add_mesh(sampled,
                                  scalars=self.cbArray.currentText(),
                                  log_scale=True,
                                  scalar_bar_args=self.sargs,
                                  cmap='jet',
                                  opacity=1.0,
                                  line_width=5.,
                                  n_colors=10)
        self.plotter.reset_camera()

    def show_prism(self):
        '''
        It generates and plots a prism using the bounds variables at the center
        of coordinates.
        Then with the help of a slider bar widget the prism can be translated
        to every point of the trajectory.
        '''
        bounds = [
                -self.bounds[0],
                +self.bounds[0],
                -self.bounds[1],
                +self.bounds[1],
                -self.bounds[2],
                +self.bounds[2]
                  ]
        t = self.trajectories[self.cbTrajectory.currentText()]
        ctraj = t.compute_traj(self.step)
        ftraj = t.flat_ctraj(ctraj)
        points = np.array([list(i) for i, z in ftraj])
        prism = pv.Cube(bounds=bounds)
        # Translate it to the initial position before the plotting to avoid
        # a plot in the origin of coordinates far from the data.
        prism.translate(points[0])
        self.plotter.add_mesh(prism,
                              color='white',
                              opacity=1.)

        def move_prism(indx):
            '''
            Translates the prism to a point of the trajectory according to indx
            which is obtained from the slider bar.
            '''
            # To use the ctraj
            vector = points[int(indx)] - np.array(prism.center)
#            To use the self.traj
#            vector = t.traj['P'][int(indx)]- np.array(prism.center)
            prism.translate(vector)
            return

        self.plotter.add_slider_widget(move_prism,
                                       [0, len(points)-1],
                                       value=0.,
                                       title='Trajectory point',
                                       event_type='always')
#        self.plotter.add_slider_widget(move_prism,
#                                       [0, len(t.traj['P'])-1],
#                                       value=0.,
#                                       title='Trajectory point')

    def clear_planes(self):
        '''
        Removes the slicing planes of the plotter. They can be annoying as
        you can click on them accidentally.
        '''
        self.plotter.clear_plane_widgets()

    def clear_plots(self):
        '''
        Clear the plotter from meshses and widgets.
        The clearing of the widgets is very important as they would remain
        in the plotter even if the mesh that created them has been removed.
        This would cause weird behaviour while clicking in the plotter as
        it encounters ghost slice planes.
        '''
        self.plotter.clear()
        self.plotter.clear_plane_widgets()
        self.plotter.clear_slider_widgets()

    def xyzbounds(self):
        '''
        It displays in a new window a table showing the XYZ bounds.
        This table can be modified to input new bounds.
        '''
        table = QTableWidget(self)
        table.setColumnCount(1)
        table.setRowCount(3)
        table.setVerticalHeaderLabels(['X', 'Y', 'Z'])
        table.setHorizontalHeaderLabels(['Value'])
        table.setItem(0, 0, QTableWidgetItem(str(self.bounds[0])))
        table.setItem(1, 0, QTableWidgetItem(str(self.bounds[1])))
        table.setItem(2, 0, QTableWidgetItem(str(self.bounds[2])))
        table.itemChanged.connect(self.changeBound)

        bDialog = QDialog(self)
        bDialog.setWindowTitle("Input XYZ bounds")
#        bDialog.setFixedSize(300, 300)
        dlayout = Qt.QVBoxLayout()
        dlayout.addWidget(table)
        dlayout.addWidget(QLabel(self, text="The prism will have as lenghts"
                                 " [x, y, z]."))
        dlayout.addWidget(QLabel(self, text="You can input different values"
                                 " directly into the table."))
        dlayout.addWidget(QLabel(self, text="If the values are 0 the"
                                 " calculations will be done considering a"
                                 " point instead of a 3D prism."))
        bDialog.setLayout(dlayout)
        bDialog.setModal(True)
        bDialog.show()

    def set_step(self):
        '''
        Modifies the self.step (Default 1)
        '''
        sDialog = QDialog(self)
        sDialog.setWindowTitle("Input step value")
        slayout = Qt.QVBoxLayout()
        table = QTableWidget(self)
        table.setColumnCount(1)
        table.setRowCount(1)
        table.setVerticalHeaderLabels([''])
        table.setHorizontalHeaderLabels(['step value'])
        table.setItem(0, 0, QTableWidgetItem(str(self.step)))
        table.itemChanged.connect(self.changeStep)
        slayout.addWidget(table)
        slayout.addWidget(QLabel(self, text="Input the step value which will "
                                 "determine the maximum\n distance between "
                                 "two computed points in a trajectory."))
        sDialog.setLayout(slayout)
        sDialog.setModal(True)
        sDialog.show()

    def csv_traj(self):
        '''
        Displays the csv trajectory table currently selected.
        '''
        t = self.trajectories[self.cbTrajectory.currentText()]
        sDialog = QDialog(self)
        sDialog.setWindowTitle(self.cbTrajectory.currentText())
        slayout = Qt.QVBoxLayout()
        table = QTableWidget(self)
        table.setColumnCount(5)
        table.setRowCount(len(t.traj['P']))
#        table.setVerticalHeaderLabels([''])
        table.setHorizontalHeaderLabels(['X', 'Y', 'Z', 'T', 'vel'])
        for i in range(len(t.traj['P'])):
            table.setItem(i, 0, QTableWidgetItem(str(t.traj['P'][i][0])))
            table.setItem(i, 1, QTableWidgetItem(str(t.traj['P'][i][1])))
            table.setItem(i, 2, QTableWidgetItem(str(t.traj['P'][i][2])))
            table.setItem(i, 3, QTableWidgetItem(str(t.traj['T'][i])))
            table.setItem(i, 4, QTableWidgetItem(str(t.traj['vel'][i])))
        slayout.addWidget(table)
        sDialog.setLayout(slayout)
        sDialog.resize(800, 500)
#        sDialog.setModal(True)
        sDialog.show()

    def dose_calculation(self):
        '''
        Writes an updated csv file with the doses along the trajectory.
        '''
        t = self.trajectories[self.cbTrajectory.currentText()]
        dose_map = self.vtk_files[self.cbVTK.currentText()]
        dose = t.calculate_dose(dose_map,
                                self.cbArray.currentText(),
                                self.step,
                                self.bounds)
        res_filename = (t.csv.split('.csv')[0]
                        + '_' + self.cbArray.currentText()
                        + '.csv')
        t.write_doses(dose, res_filename)
        QMessageBox.about(self,
                          "Dose calculation",
                          "The results have been written to " + res_filename)

    def aboutf4e(self):
        QMessageBox.about(self,
                          "Fusion for Energy",
                          "It is the EU organisation that manages the EU’s "
                          "contribution to the ITER project, designed to "
                          "demonstrate the scientific and technological "
                          "feasibility of fusion power. ")

    def aboutRPN(self):
        QMessageBox.about(self,
                          "Raul Pampin (F4E)",
                          "raul.pampin@f4e.europa.eu")

    def aboutMFI(self):
        QMessageBox.about(self,
                          "Marco Fabbri (F4E)",
                          "marco.fabbri@f4e.europa.eu")

    def aboutACI(self):
        QMessageBox.about(self,
                          "Alvaro Cubi (F4E-ext)",
                          "alvaro.cubi@ext.f4e.europa.eu")

    def aboutHCN(self):
        QMessageBox.about(self,
                          "Haridev Chohan (F4E-ext)",
                          "haridev.chohan@ext.f4e.europa.eu")

    def selectionchangeTrajectory(self):
        self.statusBar().showMessage('CSV trajectory file selected.')

    def selectionchangeVTK(self):
        self.cbArray.clear()
        self.cbArray.addItems([array for array in self.
                               vtk_files[self.cbVTK.currentText()].mesh.
                               cell_arrays.keys()])
        self.statusBar().showMessage('VTK file selected.')

    def selectionchangeArray(self):
        self.statusBar().showMessage('Data array selected.')

    def changeBound(self, item):
        '''
        Update the self.bounds with the new data from the table.
        If the data is not valid restore the table to its previous value.
        '''
        try:
            self.bounds[item.row()] = float(item.data(0))
        except Exception:
            QMessageBox.about(self,
                              "Wrong data",
                              "The value you just inputted cannot be "
                              "interpreted as a boundary.\n Please insert a "
                              "number.")

    def changeStep(self, item):
        '''
        Update the self.step with the new data from the table.
        If the data is not valid restore the table to its previous value.
        '''
        try:
            self.step = float(item.data(0))
        except Exception:
            QMessageBox.about(self, "Wrong data", "The value you just inputted"
                              " cannot be interpreted as a step.\n Please "
                              "insert a number.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
