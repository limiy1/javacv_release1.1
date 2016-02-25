/*
 * Copyright (C) 2009-2011 Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.bytedeco.javacv;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.Arrays;
import java.util.Iterator;
import java.util.LinkedList;

import org.bytedeco.javacpp.opencv_core.CvAttrList;
import org.bytedeco.javacpp.opencv_core.CvFileStorage;
import org.bytedeco.javacpp.opencv_core.FileStorage;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

/**
 *
 * @author Samuel Audet
 */
public class ProCamGeometricCalibrator {
    public ProCamGeometricCalibrator(Settings settings, MarkerDetector.Settings detectorSettings,
            MarkedPlane boardPlane, MarkedPlane projectorPlane,
            ProjectiveDevice camera, ProjectiveDevice projector) {
        this(settings, detectorSettings, boardPlane, projectorPlane, 
                new GeometricCalibrator[] { 
                new GeometricCalibrator(settings, detectorSettings, boardPlane, camera)},
                new GeometricCalibrator(settings, detectorSettings, projectorPlane, projector));
    }
    @SuppressWarnings("unchecked")
    public ProCamGeometricCalibrator(Settings settings, MarkerDetector.Settings detectorSettings,
            MarkedPlane boardPlane, MarkedPlane projectorPlane,
            GeometricCalibrator[] cameraCalibrators, GeometricCalibrator projectorCalibrator) {
        this.settings = settings;
        this.detectorSettings = detectorSettings;
        this.boardPlane = boardPlane;
        this.projectorPlane = projectorPlane;

        this.cameraCalibrators = cameraCalibrators;
        int n = cameraCalibrators.length;
        markerDetectors = new MarkerDetector[n];
        // "unchecked" warning here
        allImagedBoardMarkers = new LinkedList[n];
        grayscaleImage = new IplImage[n];
        tempImage1 = new IplImage[n];
        tempImage2 = new IplImage[n];
        lastDetectedMarkers1 = new Marker[n][];
        lastDetectedMarkers2 = new Marker[n][];
        rmseBoardWarp = new double[n];
        rmseProjWarp = new double[n];
        boardWarp = new CvMat[n];
        projWarp = new CvMat[n];
        prevBoardWarp = new CvMat[n];
        lastBoardWarp = new CvMat[n];
        tempPts1 = new CvMat[n];
        tempPts2 = new CvMat[n];
        for (int i = 0; i < n; i++) {
            markerDetectors[i] = new MarkerDetector(detectorSettings);
            allImagedBoardMarkers[i] = new LinkedList<Marker[]>();
            grayscaleImage[i] = null;
            tempImage1[i] = null;
            tempImage2[i] = null;
            lastDetectedMarkers1[i] = null;
            lastDetectedMarkers2[i] = null;
            rmseBoardWarp[i] = Double.POSITIVE_INFINITY;
            rmseProjWarp[i] = Double.POSITIVE_INFINITY;
            boardWarp[i] = CvMat.create(3, 3);
            projWarp[i] = CvMat.create(3, 3);
            prevBoardWarp[i] = CvMat.create(3, 3);
            lastBoardWarp[i] = CvMat.create(3, 3);
            cvSetIdentity(prevBoardWarp[i]);
            cvSetIdentity(lastBoardWarp[i]);
            tempPts1[i] = CvMat.create(1, 4, CV_64F, 2);
            tempPts2[i] = CvMat.create(1, 4, CV_64F, 2);
        }
        this.projectorCalibrator = projectorCalibrator;

        this.boardWarpSrcPts = CvMat.create(1, 4, CV_64F, 2);
        if (boardPlane != null) {
            int w = boardPlane.getImage().width();
            int h = boardPlane.getImage().height();
            boardWarpSrcPts.put(0.0, 0.0,  w, 0.0,  w, h,  0.0, h);
        }
        if (projectorPlane != null) {
            int w = projectorPlane.getImage().width();
            int h = projectorPlane.getImage().height();
            projectorCalibrator.getProjectiveDevice().imageWidth = w;
            projectorCalibrator.getProjectiveDevice().imageHeight = h;
        }
    }

    private final int
            MSB_IMAGE_SHIFT = 8,
            LSB_IMAGE_SHIFT = 7;

    public static class Settings extends GeometricCalibrator.Settings {
        double detectedProjectorMin = 0.5;
        boolean useOnlyIntersection = true;
        double prewarpUpdateErrorMax = 0.01;

        public double getDetectedProjectorMin() {
            return detectedProjectorMin;
        }
        public void setDetectedProjectorMin(double detectedProjectorMin) {
            this.detectedProjectorMin = detectedProjectorMin;
        }

        public boolean isUseOnlyIntersection() {
            return useOnlyIntersection;
        }
        public void setUseOnlyIntersection(boolean useOnlyIntersection) {
            this.useOnlyIntersection = useOnlyIntersection;
        }

        public double getPrewarpUpdateErrorMax() {
            return prewarpUpdateErrorMax;
        }
        public void setPrewarpUpdateErrorMax(double prewarpUpdateErrorMax) {
            this.prewarpUpdateErrorMax = prewarpUpdateErrorMax;
        }
    }

    private Settings settings;
    private MarkerDetector.Settings detectorSettings;

    // (possibly) multiple camera stuff in arrays
    private GeometricCalibrator[] cameraCalibrators;
    private MarkerDetector[] markerDetectors;
    // keep our own list of markers for the camera, since cameraCalibrators
    // might be used outside ProCamGeometricCalibrator as well...
    LinkedList<Marker[]>[] allImagedBoardMarkers;
    private IplImage[] grayscaleImage, tempImage1, tempImage2;
    private Marker[][] lastDetectedMarkers1, lastDetectedMarkers2;
    private double[] rmseBoardWarp, rmseProjWarp;
    private CvMat[] boardWarp, projWarp;
    private CvMat[] prevBoardWarp, lastBoardWarp;
    private CvMat[] tempPts1, tempPts2;

    // single board and projector stuff
    private boolean updatePrewarp = false;
    private final MarkedPlane boardPlane, projectorPlane;
    private final GeometricCalibrator projectorCalibrator;
    private final CvMat boardWarpSrcPts;

    public MarkedPlane getBoardPlane() {
        return boardPlane;
    }
    public MarkedPlane getProjectorPlane() {
        return projectorPlane;
    }
    public GeometricCalibrator[] getCameraCalibrators() {
        return cameraCalibrators;
    }
    public GeometricCalibrator getProjectorCalibrator() {
        return projectorCalibrator;
    }
    public int getImageCount() {
        int n = projectorCalibrator.getImageCount()/cameraCalibrators.length;
        for (GeometricCalibrator c : cameraCalibrators) {
            assert(c.getImageCount() == n);
        }
        return n;
    }

    public Marker[][] processCameraImage(IplImage cameraImage) {
        return processCameraImage(cameraImage, 0);
    }
    public Marker[][] processCameraImage(IplImage cameraImage, final int cameraNumber) {
        cameraCalibrators[cameraNumber].getProjectiveDevice().imageWidth = cameraImage.width();
        cameraCalibrators[cameraNumber].getProjectiveDevice().imageHeight = cameraImage.height();

        if (cameraImage.nChannels() > 1) {
            if (grayscaleImage[cameraNumber] == null ||
                    grayscaleImage[cameraNumber].width()  != cameraImage.width()  ||
                    grayscaleImage[cameraNumber].height() != cameraImage.height() ||
                    grayscaleImage[cameraNumber].depth()  != cameraImage.depth()) {
                grayscaleImage[cameraNumber] = IplImage.create(cameraImage.width(),
                        cameraImage.height(), cameraImage.depth(), 1, cameraImage.origin());
            }
            cvCvtColor(cameraImage, grayscaleImage[cameraNumber], CV_BGR2GRAY);
        } else {
            grayscaleImage[cameraNumber] = cameraImage;
        }

        final boolean boardWhiteMarkers = boardPlane.getForegroundColor().magnitude() >
                                          boardPlane.getBackgroundColor().magnitude();
        final boolean projWhiteMarkers = projectorPlane.getForegroundColor().magnitude() >
                                         projectorPlane.getBackgroundColor().magnitude();
        if (grayscaleImage[cameraNumber].depth() > 8) {
            if (tempImage1[cameraNumber] == null ||
                    tempImage1[cameraNumber].width()  != grayscaleImage[cameraNumber].width()  ||
                    tempImage1[cameraNumber].height() != grayscaleImage[cameraNumber].height()) {
                tempImage1[cameraNumber] = IplImage.create(grayscaleImage[cameraNumber].width(),
                        grayscaleImage[cameraNumber].height(), IPL_DEPTH_8U, 1,
                        grayscaleImage[cameraNumber].origin());
                tempImage2[cameraNumber] = IplImage.create(grayscaleImage[cameraNumber].width(),
                        grayscaleImage[cameraNumber].height(), IPL_DEPTH_8U, 1,
                        grayscaleImage[cameraNumber].origin());
            }
            Parallel.run(new Runnable() { public void run() {
                cvConvertScale(grayscaleImage[cameraNumber],
                        tempImage1[cameraNumber], 1.0/(1<<LSB_IMAGE_SHIFT), 0);
                lastDetectedMarkers1[cameraNumber] = cameraCalibrators[cameraNumber].
                        markerDetector.detect(tempImage1[cameraNumber], boardWhiteMarkers);
            }}, new Runnable() { public void run() {
                cvConvertScale(grayscaleImage[cameraNumber],
                        tempImage2[cameraNumber], 1.0/(1<<MSB_IMAGE_SHIFT), 0);
                lastDetectedMarkers2[cameraNumber] = 
                        markerDetectors[cameraNumber].detect(tempImage2[cameraNumber], projWhiteMarkers);
            }});
        } else {
            Parallel.run(new Runnable() { public void run() {
                lastDetectedMarkers1[cameraNumber] = cameraCalibrators[cameraNumber].
                        markerDetector.detect(grayscaleImage[cameraNumber], boardWhiteMarkers);
            }}, new Runnable() { public void run() {
                lastDetectedMarkers2[cameraNumber] = 
                        markerDetectors[cameraNumber].detect(grayscaleImage[cameraNumber], projWhiteMarkers);
            }});
        }

        return processMarkers(cameraNumber) ? 
            new Marker[][] { lastDetectedMarkers1[cameraNumber],
                             lastDetectedMarkers2[cameraNumber] } : null;
    }

    public void drawMarkers(IplImage image) {
        drawMarkers(image, 0);
    }
    public void drawMarkers(IplImage image, int cameraNumber) {
        cameraCalibrators[cameraNumber].
                markerDetector.draw(image, lastDetectedMarkers1[cameraNumber]);
        projectorCalibrator.
                markerDetector.draw(image, lastDetectedMarkers2[cameraNumber]);
    }

    public boolean processMarkers() {
        return processMarkers(0);
    }
    public boolean processMarkers(int cameraNumber) {
        return processMarkers(lastDetectedMarkers1[cameraNumber],
                lastDetectedMarkers2[cameraNumber], cameraNumber);
    }
    public boolean processMarkers(Marker[] imagedBoardMarkers, Marker[] imagedProjectorMarkers) {
        return processMarkers(imagedBoardMarkers, imagedProjectorMarkers, 0);
    }
    public boolean processMarkers(Marker[] imagedBoardMarkers, 
            Marker[] imagedProjectorMarkers, int cameraNumber) {
        rmseBoardWarp[cameraNumber] = boardPlane    .getTotalWarp(imagedBoardMarkers,     boardWarp[cameraNumber]);
        rmseProjWarp [cameraNumber] = projectorPlane.getTotalWarp(imagedProjectorMarkers, projWarp[cameraNumber]);

        int imageSize = (cameraCalibrators[cameraNumber].getProjectiveDevice().imageWidth+
                         cameraCalibrators[cameraNumber].getProjectiveDevice().imageHeight)/2;
        if (rmseBoardWarp[cameraNumber] <= settings.prewarpUpdateErrorMax*imageSize &&
                rmseProjWarp[cameraNumber] <= settings.prewarpUpdateErrorMax*imageSize) {
            updatePrewarp = true;
        } else {
            // not detected accurately enough..
            return false;
        }

        // First, check if we detected enough markers...
        if (imagedBoardMarkers    .length < boardPlane    .getMarkers().length*settings.detectedBoardMin ||
            imagedProjectorMarkers.length < projectorPlane.getMarkers().length*settings.detectedProjectorMin) {
                return false;
        }

        // then check by how much the corners of the calibration board moved
        cvPerspectiveTransform  (boardWarpSrcPts,        tempPts1[cameraNumber], boardWarp[cameraNumber]);
        cvPerspectiveTransform  (boardWarpSrcPts,        tempPts2[cameraNumber], prevBoardWarp[cameraNumber]);
        double rmsePrev = cvNorm(tempPts1[cameraNumber], tempPts2[cameraNumber]);
        cvPerspectiveTransform  (boardWarpSrcPts,        tempPts2[cameraNumber], lastBoardWarp[cameraNumber]);
        double rmseLast = cvNorm(tempPts1[cameraNumber], tempPts2[cameraNumber]);

//System.out.println("rmsePrev = " + rmsePrev + " rmseLast = " + rmseLast + " cameraNumber = " + cameraNumber);
        // save boardWarp for next iteration...
        cvCopy(boardWarp[cameraNumber], prevBoardWarp[cameraNumber]);

        // send upstream our recommendation for addition or not of these markers...
        if (rmsePrev < settings.patternSteadySize*imageSize &&
                rmseLast > settings.patternMovedSize*imageSize) {
            return true;
        } else {
            return false;
        }
    }

    public void addMarkers() {
        addMarkers(0);
    }
    public void addMarkers(int cameraNumber) {
        addMarkers(lastDetectedMarkers1[cameraNumber], lastDetectedMarkers2[cameraNumber], cameraNumber);
    }
    public void addMarkers(Marker[] imagedBoardMarkers, Marker[] imagedProjectorMarkers) {
        addMarkers(imagedBoardMarkers, imagedProjectorMarkers, 0);
    }
    private static ThreadLocal<CvMat>
            tempWarp3x3 = CvMat.createThreadLocal(3, 3);
    public void addMarkers(Marker[] imagedBoardMarkers, 
            Marker[] imagedProjectorMarkers, int cameraNumber) {
        CvMat tempWarp = tempWarp3x3.get();

        if (!settings.useOnlyIntersection) {
            cameraCalibrators[cameraNumber].addMarkers(boardPlane.getMarkers(),
                    imagedBoardMarkers);
            allImagedBoardMarkers[cameraNumber].add(imagedBoardMarkers);
        } else {
            // deep cloning... warp board markers in projector plane
            Marker[] inProjectorBoardMarkers = new Marker[imagedBoardMarkers.length];
            for (int i = 0; i < inProjectorBoardMarkers.length; i++) {
                inProjectorBoardMarkers[i] = imagedBoardMarkers[i].clone();
            }
            cvInvert(projWarp[cameraNumber], tempWarp);
            Marker.applyWarp(inProjectorBoardMarkers, tempWarp);

            // only add markers that are within the projector plane as well
            int w = projectorPlane.getImage().width();
            int h = projectorPlane.getImage().height();
            Marker[] boardMarkersToAdd = new Marker[imagedBoardMarkers.length];
            int totalToAdd = 0;
            for (int i = 0; i < inProjectorBoardMarkers.length; i++) {
                double[] c = inProjectorBoardMarkers[i].corners;
                boolean outside = false;
                for (int j = 0; j < 4; j++) {
                    int margin = detectorSettings.subPixelWindow/2;
                    if (c[2*j  ] < margin || c[2*j  ] >= w-margin ||
                        c[2*j+1] < margin || c[2*j+1] >= h-margin) {
                            outside = true;
                            break;
                    }
                }
                if (!outside) {
                    boardMarkersToAdd[totalToAdd++] = imagedBoardMarkers[i];
                }
            }
            Marker[] a = Arrays.copyOf(boardMarkersToAdd, totalToAdd);
            cameraCalibrators[cameraNumber].addMarkers(boardPlane.getMarkers(), a);
            allImagedBoardMarkers[cameraNumber].add(a);
        }

        // deep cloning...
        Marker[] prewrappedProjMarkers = new Marker[projectorPlane.getMarkers().length];
        for (int i = 0; i < prewrappedProjMarkers.length; i++) {
            prewrappedProjMarkers[i] = projectorPlane.getMarkers()[i].clone();
        }
        // prewarp points for the projectorCalibrator
        Marker.applyWarp(prewrappedProjMarkers, projectorPlane.getPrewarp());
        synchronized (projectorCalibrator) {
            // wait our turn to add markers orderly in the projector calibrator...
            while (projectorCalibrator.getImageCount()%cameraCalibrators.length < cameraNumber) {
                try {
                    projectorCalibrator.wait();
                } catch (InterruptedException ex) { }
            }
            projectorCalibrator.addMarkers(imagedProjectorMarkers, prewrappedProjMarkers);
            projectorCalibrator.notify();
        }

        // we added the detected markers, so save last computed warp too...
        cvCopy(boardWarp[cameraNumber], lastBoardWarp[cameraNumber]);
    }

    public IplImage getProjectorImage() {
        if (updatePrewarp) {
            // find which camera has the smallest RMSE
            double minRmse = Double.MAX_VALUE;
            int minCameraNumber = 0;
            for (int i = 0; i < cameraCalibrators.length; i++) {
                double rmse = rmseBoardWarp[i]+rmseProjWarp[i];
                if (rmse < minRmse) {
                    minRmse = rmse;
                    minCameraNumber = i;
                }
            }

            // and use it to update the prewarp...
            CvMat prewarp = projectorPlane.getPrewarp();
            cvInvert(          projWarp[minCameraNumber], prewarp);
            cvMatMul(prewarp, boardWarp[minCameraNumber], prewarp);
            projectorPlane.setPrewarp(prewarp);
        }

        return projectorPlane.getImage();
    }

    
    //Added by Liming
    double[] setCornerByCenter(double x, double y)
    {

    	double[] corners = new double[8];
        corners[0] = x; corners[1] = y;
        corners[4] = x; corners[5] = y;
        corners[2] = x; corners[3] = y;
        corners[6] = x; corners[7] = y;
        
        return corners;
    }
    
    boolean loadCornerData(String filename, final GeometricCalibrator cameraCalibrator, final GeometricCalibrator projectorCalibrator) throws NumberFormatException, IOException
    {
    	File file = new File(filename);
    	BufferedReader reader = null;
        reader = new BufferedReader(new FileReader(file));
        String text = null;

        int viewNum;
    	LinkedList<Marker[]>  allDistortedProjectedMarkersViewByCamera = new LinkedList<Marker[]>(),
    			allProjectorImageMarkers = new LinkedList<Marker[]>(),
    		    allDistortedImagedBoardMarkers = new LinkedList<Marker[]>(),
    		    allBoardObjectMarkers = new LinkedList<Marker[]>();
    		    
    	System.out.println("Start loading rawdata ...");
        if ((text = reader.readLine()) != null)
        {
        	viewNum = Integer.parseInt(text);
        }
        else
        {
        	reader.close();
        	return false;
        }
        
        System.out.println("View number = " + viewNum);
        int pointnum, checksum;
//        double[] refCenter = new double[2];
//        double[] imagedCenter = new double[2];
        double[] refCorner = new double[8];
        double[] imgCorner = new double[8];
        for (int i = 0 ; i < viewNum ; i++)
        {
        	//For imaged board points
            if ((text = reader.readLine()) != null)
            {
            	String[] numbers = text.split(" ");
            	pointnum = Integer.parseInt(numbers[0]);
            	checksum = Integer.parseInt(numbers[1]);
            	
            	System.out.println("pointnum = " + pointnum);
            }
            else
            {
            	reader.close();
            	return false;
            }
            
    		if (checksum != i*10 + 0)
    		{
    			System.out.println("The check sum is not correct: " + checksum + "!=" + (i*10+0));
    			reader.close();
    			return false;
    		}
    		
        	Marker[] cameraObjectMarkers = new Marker[pointnum];
        	Marker[] cameraImageMarkers = new Marker[pointnum];
    		
            for (int j = 0 ; j < pointnum ; j++)
            {
                if ((text = reader.readLine()) != null)
                {
                	//read centers
                	String[] numbers = text.split(" ");
                	for (int k = 0 ; k < 8 ; k++)
                	{
                		refCorner[k] = Double.parseDouble(numbers[k]);
                	}
                	
                	for (int k = 0 ; k < 8 ; k++)
                	{
                		imgCorner[k] = Double.parseDouble(numbers[k+8]);
                	}
                	
                	cameraObjectMarkers[j] = new Marker(j, refCorner.clone());
                	cameraImageMarkers[j] = new Marker(j, imgCorner.clone());
                }
                else
                {
                	reader.close();
                	return false;
                }
            }
            
            //For imaged projected points
            if ((text = reader.readLine()) != null)
            {
            	String[] numbers = text.split(" ");
            	pointnum = Integer.parseInt(numbers[0]);
            	checksum = Integer.parseInt(numbers[1]);
            }
            else
            {
            	reader.close();
            	return false;
            }
            
    		if (checksum != i*10 + 1)
    		{
    			System.out.println("The check sum is not correct: " + checksum + "!=" + (i*10+0));
    			reader.close();
    			return false;
    		}
    		
        	Marker[] projectorImageMarkers = new Marker[pointnum];
        	Marker[] projectedMarkersViewedByCam = new Marker[pointnum];
    		
            for (int j = 0 ; j < pointnum ; j++)
            {
                if ((text = reader.readLine()) != null)
                {
                	//read centers
                	String[] numbers = text.split(" ");
                	for (int k = 0 ; k < 8 ; k++)
                	{
                		refCorner[k] = Double.parseDouble(numbers[k]);
                	}
                	
                	for (int k = 0 ; k < 8 ; k++)
                	{
                		imgCorner[k] = Double.parseDouble(numbers[k+8]);
                	}
                	
                	//create markers
                	projectorImageMarkers[j] = new Marker(j, refCorner.clone());
                	projectedMarkersViewedByCam[j] = new Marker(j, imgCorner.clone());
                }
                else
                {
                	reader.close();
                	return false;
                }
            }
            
            //Add into the list
        	allDistortedProjectedMarkersViewByCamera.add(projectedMarkersViewedByCam);
        	allProjectorImageMarkers.add(projectorImageMarkers);
        	allDistortedImagedBoardMarkers.add(cameraImageMarkers);
        	allBoardObjectMarkers.add(cameraObjectMarkers);
        }

        reader.close();
        
        //Set into the camera and projector
    	projectorCalibrator.setAllObjectMarkers(allDistortedProjectedMarkersViewByCamera);
    	projectorCalibrator.setAllImageMarkers(allProjectorImageMarkers);
    	cameraCalibrator.setAllImageMarkers(allDistortedImagedBoardMarkers);
    	cameraCalibrator.setAllObjectMarkers(allBoardObjectMarkers);
        
    	System.out.println("Finish loading rawdata !");
    	return true;
    }

    boolean loadCenterData(String filename, final GeometricCalibrator cameraCalibrator, final GeometricCalibrator projectorCalibrator) throws NumberFormatException, IOException
    {
    	File file = new File(filename);
    	BufferedReader reader = null;
        reader = new BufferedReader(new FileReader(file));
        String text = null;

        int viewNum;
    	LinkedList<Marker[]>  allDistortedProjectedMarkersViewByCamera = new LinkedList<Marker[]>(),
    			allProjectorImageMarkers = new LinkedList<Marker[]>(),
    		    allDistortedImagedBoardMarkers = new LinkedList<Marker[]>(),
    		    allBoardObjectMarkers = new LinkedList<Marker[]>();
    		    
    	System.out.println("Start loading rawdata ...");
        if ((text = reader.readLine()) != null)
        {
        	viewNum = Integer.parseInt(text);
        }
        else
        {
        	reader.close();
        	return false;
        }
        
        System.out.println("View number = " + viewNum);
        int pointnum, checksum;
        double[] refCenter = new double[2];
        double[] imagedCenter = new double[2];
        for (int i = 0 ; i < viewNum ; i++)
        {
        	//For imaged board points
            if ((text = reader.readLine()) != null)
            {
            	String[] numbers = text.split(" ");
            	pointnum = Integer.parseInt(numbers[0]);
            	checksum = Integer.parseInt(numbers[1]);
            	
            	System.out.println("pointnum = " + pointnum);
            }
            else
            {
            	reader.close();
            	return false;
            }
            
    		if (checksum != i*10 + 0)
    		{
    			System.out.println("The check sum is not correct: " + checksum + "!=" + (i*10+0));
    			reader.close();
    			return false;
    		}
    		
        	Marker[] cameraObjectMarkers = new Marker[pointnum];
        	Marker[] cameraImageMarkers = new Marker[pointnum];
    		
            for (int j = 0 ; j < pointnum ; j++)
            {
                if ((text = reader.readLine()) != null)
                {
                	//read centers
                	String[] numbers = text.split(" ");
                	
                	refCenter[0] = Double.parseDouble(numbers[0]);
                	refCenter[1] = Double.parseDouble(numbers[1]);         	
                	imagedCenter[0] = Double.parseDouble(numbers[2]);
                	imagedCenter[1] = Double.parseDouble(numbers[3]);          	
                	System.out.println("read : " + Arrays.toString(refCenter) + " " + Arrays.toString(imagedCenter));
                	
                	//create markers
                	cameraObjectMarkers[j] = new Marker(j, setCornerByCenter(refCenter[0], refCenter[1]));
                	cameraImageMarkers[j] = new Marker(j, setCornerByCenter(imagedCenter[0], imagedCenter[1]));
                }
                else
                {
                	reader.close();
                	return false;
                }
            }
            
            //For imaged projected points
            if ((text = reader.readLine()) != null)
            {
            	String[] numbers = text.split(" ");
            	pointnum = Integer.parseInt(numbers[0]);
            	checksum = Integer.parseInt(numbers[1]);
            }
            else
            {
            	reader.close();
            	return false;
            }
            
    		if (checksum != i*10 + 1)
    		{
    			System.out.println("The check sum is not correct: " + checksum + "!=" + (i*10+0));
    			reader.close();
    			return false;
    		}
    		
        	Marker[] projectorImageMarkers = new Marker[pointnum];
        	Marker[] projectedMarkersViewedByCam = new Marker[pointnum];
    		
            for (int j = 0 ; j < pointnum ; j++)
            {
                if ((text = reader.readLine()) != null)
                {
                	//read centers
                	String[] numbers = text.split(" ");
                	
                	refCenter[0] = Double.parseDouble(numbers[0]);
                	refCenter[1] = Double.parseDouble(numbers[1]);
                	imagedCenter[0] = Double.parseDouble(numbers[2]);
                	imagedCenter[1] = Double.parseDouble(numbers[3]);
                	System.out.println("read : " + Arrays.toString(refCenter) + " " + Arrays.toString(imagedCenter));
                	
                	//create markers
              	
                	projectorImageMarkers[j] = new Marker(j, setCornerByCenter(refCenter[0], refCenter[1]));
                	projectedMarkersViewedByCam[j] = new Marker(j, setCornerByCenter(imagedCenter[0], imagedCenter[1]));
                }
                else
                {
                	reader.close();
                	return false;
                }
            }
            
            //Add into the list
        	allDistortedProjectedMarkersViewByCamera.add(projectedMarkersViewedByCam);
        	allProjectorImageMarkers.add(projectorImageMarkers);
        	allDistortedImagedBoardMarkers.add(cameraImageMarkers);
        	allBoardObjectMarkers.add(cameraObjectMarkers);
        }

        reader.close();
        
        //Set into the camera and projector
    	projectorCalibrator.setAllObjectMarkers(allDistortedProjectedMarkersViewByCamera);
    	projectorCalibrator.setAllImageMarkers(allProjectorImageMarkers);
    	cameraCalibrator.setAllImageMarkers(allDistortedImagedBoardMarkers);
    	cameraCalibrator.setAllObjectMarkers(allBoardObjectMarkers);
        
    	System.out.println("Finish loading rawdata !");
    	return true;
    }
    
    public void saveCornerData(String filename, final GeometricCalibrator cameraCalibrator, final GeometricCalibrator projectorCalibrator)
    {
    	System.out.println("Start to print rawdata ...");
    	
    	PrintWriter writer = null;
    	try {
			writer = new PrintWriter(filename, "UTF-8");
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    	
    	System.out.println("Start writing rawdata file ...");
    	LinkedList<Marker[]>  allDistortedProjectedMarkersViewByCamera = projectorCalibrator.getAllObjectMarkers(),
    			allProjectorImageMarkers = projectorCalibrator.getAllImageMarkers(),
    		    allDistortedImagedBoardMarkers = cameraCalibrator.getAllImageMarkers(),
    		    allBoardObjectMarkers = cameraCalibrator.getAllObjectMarkers();
    			
    	Iterator<Marker[]> itCameraObjectMarker = allBoardObjectMarkers.iterator();
    	Iterator<Marker[]> itCameraImageMarker = allDistortedImagedBoardMarkers.iterator();
    	Iterator<Marker[]> itProjectedMarkerViewedByCamera = allDistortedProjectedMarkersViewByCamera.iterator();  //projectedPattern view by the camera
    	Iterator<Marker[]> itProjectorImageMarker = allProjectorImageMarkers.iterator();   //source projected pattern
    	
    	writer.println(allBoardObjectMarkers.size());
    	
    	int viewNum = 0;
    	int pointnum, checksum;
    	while(itCameraObjectMarker.hasNext())
    	{
            Marker[] cameraObjectMarkers = itCameraObjectMarker.next(),
                     cameraImageMarkers = itCameraImageMarker.next(),
                     projectedMarkersViewedByCam = itProjectedMarkerViewedByCamera.next(),
                     projectorImageMarkers = itProjectorImageMarker.next();
            
            pointnum = cameraObjectMarkers.length; checksum = viewNum*10 + 0;
    		writer.println(pointnum + " " + checksum);
            for (int i = 0; i < cameraObjectMarkers.length; i++) {
            	double [] objectCorner = cameraObjectMarkers[i].corners;
            	double [] imageCorner = cameraImageMarkers[i].corners;
            	writer.println(
            			objectCorner[0] + " " + objectCorner[1] + " " 
            			+ objectCorner[2] + " " + objectCorner[3] + " "
            			+ objectCorner[4] + " " + objectCorner[5] + " "
            			+ objectCorner[6] + " " + objectCorner[7] + " "
            			+ imageCorner[0] + " " + imageCorner[1] + " "
            			+ imageCorner[2] + " " + imageCorner[3] + " "
            			+ imageCorner[4] + " " + imageCorner[5] + " "
            			+ imageCorner[6] + " " + imageCorner[7]);
            }
            
            pointnum = projectedMarkersViewedByCam.length; checksum = viewNum*10 + 1;
            writer.println(pointnum + " " + checksum);
            for (int i = 0; i < projectedMarkersViewedByCam.length; i++) {
            	double [] objectCorner = projectedMarkersViewedByCam[i].corners;
            	double [] imageCorner = projectorImageMarkers[i].corners;
            	//Note: we write the marker center in the projector image first
            	writer.println(
            			imageCorner[0] + " " + imageCorner[1] + " "
            			+ imageCorner[2] + " " + imageCorner[3] + " "
            			+ imageCorner[4] + " " + imageCorner[5] + " "
            			+ imageCorner[6] + " " + imageCorner[7] + " "
            			+ objectCorner[0] + " " + objectCorner[1] + " " 
            			+ objectCorner[2] + " " + objectCorner[3] + " "
            			+ objectCorner[4] + " " + objectCorner[5] + " "
            			+ objectCorner[6] + " " + objectCorner[7]);
            }
            viewNum++;
    	}
    	
    	writer.close();
    	
    	System.out.println("Finish printing rawdata !");
    }
    
    public void saveCenterData(String filename, final GeometricCalibrator cameraCalibrator, final GeometricCalibrator projectorCalibrator)
    {
    	System.out.println("Start to print rawdata ...");
    	
    	PrintWriter writer = null;
    	try {
			writer = new PrintWriter(filename, "UTF-8");
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    	
    	System.out.println("Start writing rawdata file ...");
    	LinkedList<Marker[]>  allDistortedProjectedMarkersViewByCamera = projectorCalibrator.getAllObjectMarkers(),
    			allProjectorImageMarkers = projectorCalibrator.getAllImageMarkers(),
    		    allDistortedImagedBoardMarkers = cameraCalibrator.getAllImageMarkers(),
    		    allBoardObjectMarkers = cameraCalibrator.getAllObjectMarkers();
    			
    	Iterator<Marker[]> itCameraObjectMarker = allBoardObjectMarkers.iterator();
    	Iterator<Marker[]> itCameraImageMarker = allDistortedImagedBoardMarkers.iterator();
    	Iterator<Marker[]> itProjectedMarkerViewedByCamera = allDistortedProjectedMarkersViewByCamera.iterator();  //projectedPattern view by the camera
    	Iterator<Marker[]> itProjectorImageMarker = allProjectorImageMarkers.iterator();   //source projected pattern
    	
    	writer.println(allBoardObjectMarkers.size());
    	
    	int viewNum = 0;
    	int pointnum, checksum;
    	while(itCameraObjectMarker.hasNext())
    	{
            Marker[] cameraObjectMarkers = itCameraObjectMarker.next(),
                     cameraImageMarkers = itCameraImageMarker.next(),
                     projectedMarkersViewedByCam = itProjectedMarkerViewedByCamera.next(),
                     projectorImageMarkers = itProjectorImageMarker.next();
            
            pointnum = cameraObjectMarkers.length; checksum = viewNum*10 + 0;
    		writer.println(pointnum + " " + checksum);
            for (int i = 0; i < cameraObjectMarkers.length; i++) {
            	double [] objectCenter = cameraObjectMarkers[i].getCenter();
            	double [] imageCenter = cameraImageMarkers[i].getCenter();
            	writer.println(objectCenter[0] + " " + objectCenter[1] + " " + imageCenter[0] + " " + imageCenter[1]);
            }
            
            pointnum = projectedMarkersViewedByCam.length; checksum = viewNum*10 + 1;
            writer.println(pointnum + " " + checksum);
            for (int i = 0; i < projectedMarkersViewedByCam.length; i++) {
            	double [] objectCenter = projectedMarkersViewedByCam[i].getCenter();
            	double [] imageCenter = projectorImageMarkers[i].getCenter();
            	//Note: we write the marker center in the projector image first
            	writer.println(imageCenter[0] + " " + imageCenter[1] + " " + objectCenter[0] + " " + objectCenter[1]);
            }
            viewNum++;
    	}
    	
    	writer.close();
    	
    	System.out.println("Finish printing rawdata !");
    }
    
    public double[] calibrate(boolean useCenters, boolean calibrateCameras) {
        return calibrate(useCenters, calibrateCameras);
    }
    
    @SuppressWarnings("unchecked")
    public double[] calibrate(boolean useCenters, boolean calibrateCameras, int cameraAtOrigin) {
        GeometricCalibrator calibratorAtOrigin = cameraCalibrators[cameraAtOrigin];

        //Liming added
        {
        	System.out.println("Start to register rawdata (cameraCalibrators.length = " + cameraCalibrators.length + ") ...");
	        if (cameraCalibrators.length == 1) {
	        	try {
					loadCenterData("input_rawdata.txt", cameraCalibrators[0], projectorCalibrator);
				} catch (NumberFormatException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
	        	
	        	saveCenterData("rawdatatext.txt", cameraCalibrators[0], projectorCalibrator);
	        }
	        else
	        {
	        	System.out.println("Problem in registering rawdata! ");
	        }
        }
        
    	System.out.println("Camera : " + cameraCalibrators[0].getAllObjectMarkers().size() + ", " + cameraCalibrators[0].getAllImageMarkers().size());
    	System.out.println("Projector : " + projectorCalibrator.getAllObjectMarkers().size() + ", " + projectorCalibrator.getAllImageMarkers().size());
        //End added
    	
        // calibrate camera if not already calibrated...
        if (calibrateCameras) {
            for (int cameraNumber = 0; cameraNumber < cameraCalibrators.length; cameraNumber++) {
            	System.out.println("Now calibrate camera " + cameraNumber);
                cameraCalibrators[cameraNumber].calibrate(useCenters);
                if (cameraCalibrators[cameraNumber] != calibratorAtOrigin) {
                    calibratorAtOrigin.calibrateStereo(useCenters, cameraCalibrators[cameraNumber]);
                }
            }
        }
        
        // remove distortion from corners of imaged markers for projector calibration
        // (in the case of the projector, markers imaged by the cameras, that is
        // those affected by their distortions, are the "object" markers, but
        // we need to remove this distortion, something we can do now that we
        // have calibrated the cameras...)
        LinkedList<Marker[]> allDistortedProjectorMarkers = projectorCalibrator.getAllObjectMarkers(),
                             distortedProjectorMarkersAtOrigin = new LinkedList<Marker[]>(),
                             allUndistortedProjectorMarkers = new LinkedList<Marker[]>(),
                             undistortedProjectorMarkersAtOrigin = new LinkedList<Marker[]>();
                             
        Iterator<Marker[]> ip = allDistortedProjectorMarkers.iterator();
        // "unchecked" warning here
        Iterator<Marker[]>[] ib = new Iterator[cameraCalibrators.length],
        		ibobj = new Iterator[cameraCalibrators.length];
        for (int cameraNumber = 0; cameraNumber < cameraCalibrators.length; cameraNumber++) {
//            ib[cameraNumber] = allImagedBoardMarkers[cameraNumber].iterator();   //Liming: Problem with allImageBoardMarkers here
        	ib[cameraNumber] = cameraCalibrators[cameraNumber].getAllImageMarkers().iterator(); 
        	ibobj[cameraNumber] = cameraCalibrators[cameraNumber].getAllObjectMarkers().iterator();
        }

        // iterate over all the saved markers in the right order...
        // eew, this is getting ugly...
        while (ip.hasNext()) {
            for (int cameraNumber = 0; cameraNumber < cameraCalibrators.length; cameraNumber++) {
                double maxError = settings.prewarpUpdateErrorMax *
                        (cameraCalibrators[cameraNumber].getProjectiveDevice().imageWidth+
                         cameraCalibrators[cameraNumber].getProjectiveDevice().imageHeight)/2;
                
                Marker[] boardMarkers = ibobj[cameraNumber].next(),
                		 distortedBoardMarkers = ib[cameraNumber].next(),
                         distortedProjectorMarkers = ip.next(),
                         undistortedBoardMarkers = new Marker[distortedBoardMarkers.length],
                         undistortedProjectorMarkers = new Marker[distortedProjectorMarkers.length];

                // remove radial distortion from all points imaged by the camera
                for (int i = 0; i < distortedBoardMarkers.length; i++) {
                    Marker m = undistortedBoardMarkers[i] = distortedBoardMarkers[i].clone();
                    m.corners = cameraCalibrators[cameraNumber].getProjectiveDevice().undistort(m.corners);
                }
                for (int i = 0; i < distortedProjectorMarkers.length; i++) {
                    Marker m = undistortedProjectorMarkers[i] = distortedProjectorMarkers[i].clone();
                    m.corners = cameraCalibrators[cameraNumber].getProjectiveDevice().undistort(m.corners);
                }
                
                // Estimate the homography(boardWarp[cameraNumber]) camera-board, using unditorted board markers
                // remove linear distortion/warping of camera imaged markers from
                // the projector, to get their physical location on the board
                if (boardPlane.getTotalWarp(boardMarkers, undistortedBoardMarkers, boardWarp[cameraNumber]) > maxError) {
//                if (boardPlane.getTotalWarp(undistortedBoardMarkers, boardWarp[cameraNumber]) > maxError) {
                    assert(false);
                }
                
                //boardWarp[cameraNumber] = homography
                cvInvert(boardWarp[cameraNumber], boardWarp[cameraNumber]);
                Marker.applyWarp(undistortedProjectorMarkers, boardWarp[cameraNumber]);

                // tadam, we not have undistorted "object" corners for the projector..
                allUndistortedProjectorMarkers.add(undistortedProjectorMarkers);
                if (cameraCalibrators[cameraNumber] == calibratorAtOrigin) {
                    undistortedProjectorMarkersAtOrigin.add(undistortedProjectorMarkers);
                    distortedProjectorMarkersAtOrigin.add(distortedProjectorMarkers);
                } else {
                    undistortedProjectorMarkersAtOrigin.add(new Marker[0]);
                    distortedProjectorMarkersAtOrigin.add(new Marker[0]);
                }
            }
        }

        // calibrate projector
        projectorCalibrator.setAllObjectMarkers(allUndistortedProjectorMarkers);
        System.out.println("Now calibrate projector...");
        double[] reprojErr = projectorCalibrator.calibrate(useCenters);
//        projectorCalibrator.getProjectiveDevice().nominalDistance =
//                projectorCalibrator.getProjectiveDevice().getNominalDistance(boardPlane);

        // calibrate as a stereo pair (find rotation and translation)
        // let's use the projector markers only...
        LinkedList<Marker[]> om = calibratorAtOrigin.getAllObjectMarkers(),
                             im = calibratorAtOrigin.getAllImageMarkers();
        calibratorAtOrigin.setAllObjectMarkers(undistortedProjectorMarkersAtOrigin);
        calibratorAtOrigin.setAllImageMarkers(distortedProjectorMarkersAtOrigin);
        double[] epipolarErr = calibratorAtOrigin.calibrateStereo(useCenters, projectorCalibrator);

        // reset everything as it was before we started, so we get the same
        // result if called a second time..
        projectorCalibrator.setAllObjectMarkers(allDistortedProjectorMarkers);
        calibratorAtOrigin.setAllObjectMarkers(om);
        calibratorAtOrigin.setAllImageMarkers(im);

        return new double[] { reprojErr[0], reprojErr[1], epipolarErr[0], epipolarErr[1] };
    }

}
