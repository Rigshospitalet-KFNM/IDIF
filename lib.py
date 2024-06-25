from skimage.measure import label
from tqdm import tqdm
from numpy import np
from pydicom import dcmread
import csv

def dcmread_folder(filelist: list):
    # Loop over all files
    nfiles = len(filelist)
    if nfiles == 0:
        raise SystemExit('no files in filelist')
    print(f'Looping over {len(filelist)} DCM files')
    for dcmfile in tqdm(filelist):
        # Read meta data of DICOM file
        ds = dcmread(dcmfile, False)
        
        # Get SeriesInstanceUID and modality
        SeriesInstanceUID = ds[0x0020, 0x000e].value
        Modality = ds.Modality
        
        if Modality == 'PT':
            # Check Series Type (0054,1000)
            SeriesType = ds[0x0054,0x1000].value[0]
            
            # Image Index
            ImageIndex = ds[0x0054,0x1330].value
            
            # Determine slice and possibly frame number from Image Index
            if SeriesType == 'DYNAMIC':
                # Get number of slices
                NumberOfSlices = ds[0x0054,0x0081].value
                NumberOfFrames = len(filelist)//NumberOfSlices
                
                # FROM DICOM MANUAL
                # ImageIndex = ((Time Slice Index - 1) * (NumberOfSlices)) + Slice Index
                Frame, Slice = divmod(ImageIndex-1,NumberOfSlices)
                Frame += 1
                Slice += 1
            
            elif SeriesType in ['STATIC','WHOLE BODY']:
                Slice = ImageIndex
                Frame = 1
                NumberOfSlices = len(filelist)
                
        # Get actual frame duration in seconds
        ActualFrameDuration = ds[0x0018,0x1242].value * 0.001
        if ActualFrameDuration.is_integer():
            ActualFrameDuration = int(ActualFrameDuration)
                
        if 'deck' not in locals():
            # Allocate
            deck = np.empty((ds.Rows,ds.Columns,NumberOfSlices,NumberOfFrames),dtype='float32')
            AcquisitionTime = np.empty((NumberOfFrames,), dtype='datetime64[ms]')
            FrameDuration = np.empty((NumberOfFrames,), dtype=int)
            M44 = np.empty((4,4,NumberOfSlices), dtype='float')
        
        # Get pixel data and apply zoom factor and rescaling
        deck[:,:,Slice-1, Frame-1] = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        AcquisitionTime[Frame-1] = datetime.strptime(ds.AcquisitionDate + ds.AcquisitionTime,'%Y%m%d%H%M%S.%f')
        FrameDuration[Frame-1] = ds.ActualFrameDuration
        M44[:,:,Slice-1] = affine3d(ds)

    FrameTimeStart = (AcquisitionTime - AcquisitionTime[0])//1000

    # Store additional header info
    hdr = {}
    hdr['Modality'] = ds.Modality
    hdr['Manufacturer'] = ds.Manufacturer
    hdr['ManufacturersModelName'] = ds[0x8, 0x1090].value
    hdr['InstitutionName'] = ds.InstitutionName
    hdr['InstitutionalDepartmentName'] = ds.InstitutionalDepartmentName
    hdr['InstitutionAddress'] = ds.InstitutionAddress
    hdr['DeviceSerialNumber'] = ds.DeviceSerialNumber
    hdr['StationName'] = ds.StationName
    hdr['BodyPartExamined'] = ds.get('BodyPartExamined', 'XX')
    hdr['PatientPosition'] = ds.PatientPosition
    hdr['SoftwareVersions'] = ds.SoftwareVersions
    hdr['SeriesDescription'] = ds.SeriesDescription
    hdr['ProtocolName'] = ds.ProtocolName
    hdr['ImageType'] = list(ds.ImageType)
    hdr['SeriesNumber'] = ds.SeriesNumber
    hdr['AcquisitionTime'] = ds.SeriesTime
    hdr['AcquisitionNumber'] = ds.get('AcquisitionNumber', 'XX')
    hdr['ImageComments'] = ds.ImageComments
    
    # Isotope specific fields
    hdr['Radiopharmaceutical'] = ds.RadiopharmaceuticalInformationSequence[0].Radiopharmaceutical
    hdr['RadionuclidePositronFraction'] = ds.RadiopharmaceuticalInformationSequence[0].RadionuclidePositronFraction
    hdr['RadionuclideTotalDose'] = ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose # [Bq]
    hdr['RadionuclideHalfLife'] = ds.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife # [s]
    hdr['DoseCalibrationFactor'] = ds.DoseCalibrationFactor
    hdr['ConvolutionKernel'] = ds.ConvolutionKernel
    hdr['Unit'] = ds.Units
    hdr['DecayCorrection'] = ds.DecayCorrection
    hdr['AttenuationCorrectionMethod'] = ds.AttenuationCorrectionMethod
    hdr['ReconstructionMethod'] = ds.ReconstructionMethod
    hdr['Axial Acceptance'] = ds.AxialAcceptance
    
    # hdr['DecayFactor'] = DecayFactor[frameidx].tolist()
    hdr['FrameTimesStart'] = FrameTimeStart.astype(int)
    hdr['FrameDuration'] = FrameDuration//1000
    hdr['PixelSpacing'] = np.array([float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])])
    hdr['SliceThickness'] = ds.SliceThickness
    
    # Patient specific fields
    hdr['PatientWeight'] = ds.PatientWeight # [kg]

    # Get and store transformation matrix and dimlabel
    if NumberOfSlices > 1:
        k1,k2,k3 = (M44[0:3,-1,-1]-M44[0:3,-1,0])/(NumberOfSlices-1)
        
        M44 = M44[:,:,0]
        M44[0:3,2] = [k1,k2,k3]
    else:
        M44 = affine3d(ds,NumberOfSlices=1)
    
    hdr['Affine3D'] = M44
    hdr['dimlabel'] = mat44_to_orientation(M44)
    
    return hdr, deck

def labels_to_regions():
    # Create Dict with label numbers and structures
    labels = {
        1:"spleen",
        2:"kidney_right", 	
        3:"kidney_left", 	
        4:"gallbladder", 	
        5:"liver", 	
        6:"stomach", 	
        7:"pancreas", 	
        8:"adrenal_gland_right",
        9:"adrenal_gland_left",
        10:"lung_upper_lobe_left",
        11:"lung_lower_lobe_left",
        12:"lung_upper_lobe_right",
        13:"lung_middle_lobe_right",
        14:"lung_lower_lobe_right",
        15:"esophagus", 	
        16:"trachea", 	
        17:"thyroid_gland", 	
        18:"small_bowel",
        19:"duodenum", 	
        20:"colon", 	
        21:"urinary_bladder", 	
        22:"prostate", 	
        23:"kidney_cyst_left", 	
        24:"kidney_cyst_right", 	
        25:"sacrum", 	
        26:"vertebrae_S1", 	
        27:"vertebrae_L5", 	
        28:"vertebrae_L4", 	
        29:"vertebrae_L3", 	
        30:"vertebrae_L2", 	
        31:"vertebrae_L1", 	
        32:"vertebrae_T12", 	
        33:"vertebrae_T11", 	
        34:"vertebrae_T10", 	
        35:"vertebrae_T9",	
        36:"vertebrae_T8", 	
        37:"vertebrae_T7", 	
        38:"vertebrae_T6", 	
        39:"vertebrae_T5", 	
        40:"vertebrae_T4", 	
        41:"vertebrae_T3", 	
        42:"vertebrae_T2", 	
        43:"vertebrae_T1", 	
        44:"vertebrae_C7", 	
        45:"vertebrae_C6", 	
        46:"vertebrae_C5", 	
        47:"vertebrae_C4", 	
        48:"vertebrae_C3", 	
        49:"vertebrae_C2", 	
        50:"vertebrae_C1", 	
        51:"heart", 	
        52:"aorta", 	
        53:"pulmonary_vein", 	
        54:"brachiocephalic_trunk", 	
        55:"subclavian_artery_right", 	
        56:"subclavian_artery_left", 	
        57:"common_carotid_artery_right", 	
        58:"common_carotid_artery_left", 	
        59:"brachiocephalic_vein_left", 	
        60:"brachiocephalic_vein_right", 	
        61:"atrial_appendage_left", 	
        62:"superior_vena_cava", 	
        63:"inferior_vena_cava", 	
        64:"portal_vein_and_splenic_vein",
        65:"iliac_artery_left",
        66:"iliac_artery_right",
        67:"iliac_vena_left",
        68:"iliac_vena_right",
        69:"humerus_left", 	
        70:"humerus_right", 	
        71:"scapula_left", 	
        72:"scapula_right", 	
        73:"clavicula_left",
        74:"clavicula_right",
        75:"femur_left", 	
        76:"femur_right", 	
        77:"hip_left", 	
        78:"hip_right", 	
        79:"spinal_cord", 	
        80:"gluteus_maximus_left",
        81:"gluteus_maximus_right",
        82:"gluteus_medius_left",
        83:"gluteus_medius_right",
        84:"gluteus_minimus_left",
        85:"gluteus_minimus_right",
        86:"autochthon_left", 	
        87:"autochthon_right", 	
        88:"iliopsoas_left",
        89:"iliopsoas_right",
        90:"brain", 	
        91:"skull", 	
        92:"rib_right_4", 	
        93:"rib_right_3", 	
        94:"rib_left_1", 	
        95:"rib_left_2", 	
        96:"rib_left_3", 	
        97:"rib_left_4", 	
        98:"rib_left_5", 	
        99:"rib_left_6", 	
        100:"rib_left_7", 	
        101:"rib_left_8", 	
        102:"rib_left_9", 	
        103:"rib_left_10", 	
        104:"rib_left_11", 	
        105:"rib_left_12", 	
        106:"rib_right_1", 	
        107:"rib_right_2", 	
        108:"rib_right_5", 	
        109:"rib_right_6", 	
        110:"rib_right_7", 	
        111:"rib_right_8", 	
        112:"rib_right_9", 	
        113:"rib_right_10", 	
        114:"rib_right_11", 	
        115:"rib_right_12",
        116:"sternum",
        117:"costal_cartilages"
    }
    return(labels)

def get_regionidx(region: str):
    # return label value of region
    labels = labels_to_regions()
    label = get_key_from_value(labels, region)
    return label

def get_key_from_value(d, val):
    for k,v in d.items():
        if d[k].lower() == val:
            key = k
    if not 'key' in locals():
        raise SystemExit(val + ' region not found in labelmap')
    return(key)

def bbox(mask):
    """ Returns a bounding box from binary image

    input:
      3D binary

    output:
      xmin xsize ymin ysize zmin zsize

    """
    # x_idx = np.where(np.any(mask, axis=0))[0]
    # x1, x2 = x_idx[[0, -1]]

    # y_idx = np.where(np.any(mask, axis=1))[0]
    # y1, y2 = y_idx[[0, -1]]

    # return np.array([y1,x1,y2,x2])

    imadim = mask.shape

    xmin = imadim[0]-1
    xmax = 0
    ymin = imadim[1]-1
    ymax = 0
    zmin = imadim[2]-1
    zmax = 0

    for z in range(0,imadim[2]):
        for y in range(0,imadim[1]):
            for x in range(0,imadim[0]):
                if mask[x,y,z]:
                    if x<xmin : xmin=x
                    if x>xmax : xmax=x
                    if y<ymin : ymin=y
                    if y>ymax : ymax=y
                    if z<zmin : zmin=z
                    if z>zmax : zmax=z

    return xmin,1+xmax-xmin,ymin,1+ymax-ymin,zmin,1+zmax-zmin

def dcmmlist(filelist: list, outdir: str=None):
    """ Save a tab seperated file with info of DICOM directory

    Parameters
    ----------
    filelist : string
    List of files to create mlist from

    outdir : string
    Output path where the files will be placed
    """

    # Allocate
    studydict = {}

    # Loop over all files
    for dcmfile in tqdm(filelist, desc="dcm.dcmmlist"):
        # Read meta data of DICOM file
        ds = dcmread(dcmfile,only_header=True)

        # Get SeriesInstanceUID and modality
        StudyInstanceUID = ds[0x0020, 0x000d].value
        SeriesInstanceUID = ds[0x0020, 0x000e].value
        Modality = ds.Modality

        if StudyInstanceUID not in studydict.keys():
            studydict[StudyInstanceUID] = {}

        # Create dictionary of SeriesInstanceUID
        if SeriesInstanceUID not in studydict[StudyInstanceUID]:
            # Add Series Number and Series Description to dict
            studydict[StudyInstanceUID][SeriesInstanceUID] = {
                'PatientID': ds.get('PatientID','N/A'),
                'AccessionNumber': ds.AccessionNumber,
                'StudyDescription': ds.StudyDescription.replace(' ','_'),
                'SeriesNumber': str(ds[0x0020, 0x0011].value),
                'SeriesDescription': ds[0x0008, 0x103e].value.replace(' ','_'),
                'Modality': Modality,
                'Tracer': ds.RadiopharmaceuticalInformationSequence[0].Radiopharmaceutical if 'RadiopharmaceuticalInformationSequence' in ds else 'N/A',
                'Reconstruction': ds.get('ReconstructionMethod','N/A'),
                'PixelSpacing': np.append(np.array(ds.get('PixelSpacing',np.array([0,0]))),ds.get('SliceThickness', 0)),
                'Matrix': np.array([int(ds.get('Rows',0)),int(ds.get('Columns',0)),int(ds.get('NumberOfSlices',0))]),
                'nFrames': ds.NumberOfTimeSlices if 'NumberOfTimeSlices' in ds else 1,
                'ReconFilter': ds.get('ConvolutionKernel','N/A'),
                'mlist' : []
            }

        # Get software version
        #SoftwareVersions = ds.get('SoftwareVersions','N/A')

        if Modality == 'PT':
            # Check Series Type (0054,1000)
            SeriesType = ds.get('SeriesType', 'Unknown')
            if SeriesType == 'Unknown':
                continue

            SeriesType = ds[0x0054,0x1000].value[0]

            # Image Index
            ImageIndex = ds[0x0054,0x1330].value

            # Determine slice and possibly frame number from Image Index
            if SeriesType == 'DYNAMIC':
                # Get number of slices
                NumberOfSlices = ds[0x0054,0x0081].value

                # FROM DICOM MANUAL
                # ImageIndex = ((Time Slice Index - 1) * (NumberOfSlices)) + Slice Index
                Frame, Slice = divmod(ImageIndex-1,NumberOfSlices)
                Frame += 1
                Slice += 1
            elif SeriesType in ['STATIC','WHOLE BODY','WHOLEBODY']:
                Slice = ImageIndex
                Frame = 1

            # Get actual frame duration in seconds
            ActualFrameDuration = ds[0x0018,0x1242].value * 0.001
            if ActualFrameDuration.is_integer():
                ActualFrameDuration = int(ActualFrameDuration)

        elif Modality == 'CT':
            Slice = ds.InstanceNumber
            Frame = 1
            ActualFrameDuration = 0

        # Get acquisition time (0008, 0032)
        try:
            AcquisitionTime = datetime.strptime(ds[0x0008,0x0022].value+ds[0x0008,0x0032].value,'%Y%m%d%H%M%S.%f')
        except KeyError as ke:
            continue # no AcqTime in file.

        # Get Series time (0008, 0031)
        SeriesTime = datetime.strptime(ds[0x0008,0x0021].value+ds[0x0008,0x0031].value,'%Y%m%d%H%M%S.%f')

        # Subtract to get Frame Start Time
        FrameTimeStart = (AcquisitionTime - SeriesTime).total_seconds()

        if FrameTimeStart.is_integer():
            FrameTimeStart = int(FrameTimeStart)

        # Append values to list
        studydict[StudyInstanceUID][SeriesInstanceUID]['mlist'].append([Slice, Frame, FrameTimeStart, ActualFrameDuration, AcquisitionTime.strftime('%H:%M:%S'), dcmfile])

    # Create series specific mlist files
    for StudyInstanceUID in studydict.keys():
        for SeriesInstanceUID in studydict[StudyInstanceUID].keys():
            corrupt_series = False
            # Sort list based on first Slice then Frame
            studydict[StudyInstanceUID][SeriesInstanceUID]['mlist'] = sorted(studydict[StudyInstanceUID][SeriesInstanceUID]['mlist'], key = lambda x: (x[0],x[1]))

            # CT dicom does not contain slice number info. Fingers crossed that the series is complete
            if studydict[StudyInstanceUID][SeriesInstanceUID]['Modality'] == 'CT':
                studydict[StudyInstanceUID][SeriesInstanceUID]['Matrix'][...,-1] = len(filelist)

            no_good_series = ['Statistics','Report']
            if (
                ((int(studydict[StudyInstanceUID][SeriesInstanceUID]['nFrames'])*int(studydict[StudyInstanceUID][SeriesInstanceUID]['Matrix'][...,-1]) != len(filelist))
                and studydict[StudyInstanceUID][SeriesInstanceUID]['Modality'] == 'PT')   
               ):
                if not any([string for string in no_good_series if re.findall(string,studydict[StudyInstanceUID][SeriesInstanceUID]['SeriesDescription'])]):
                    corrupt_series = True
                    duplicate = False
                    filepath = Path('/raid/logs/corrupt_series.txt')
                    if filepath.exists():
                        with open(filepath,'r') as fr:
                            for line in fr:
                                if SeriesInstanceUID in line:
                                    duplicate = True
                    else:
                        filepath.touch(exist_ok=False)
                    if not duplicate: 
                        with open(filepath,'a') as fa:
                            fa.write(studydict[StudyInstanceUID][SeriesInstanceUID]['PatientID']+':'+SeriesInstanceUID+'\n')

            if outdir and (not corrupt_series):
                #print(studydict[StudyInstanceUID][SeriesInstanceUID]['mlist'][0][2])
                if studydict[StudyInstanceUID][SeriesInstanceUID]['mlist']: # list turns out empty for some cases?
                    if studydict[StudyInstanceUID][SeriesInstanceUID]['mlist'][0][2] < 0: # we should be sorted by plane, frame, start of frame, duration at this point
                        studydict[StudyInstanceUID][SeriesInstanceUID]['mlist'] = [[plane, frame, int(framestarttime)+1, dur, acqtime, dcmfile] for plane, frame, framestarttime, dur, acqtime, dcmfile in studydict[StudyInstanceUID][SeriesInstanceUID]['mlist']]

                # Create output path
                output = os.path.join(outdir,studydict[StudyInstanceUID][SeriesInstanceUID]['PatientID'],StudyInstanceUID)

                if not os.path.isdir(output):
                    os.makedirs(output)

                # Headers
                fieldnames = ['Plane','Frame','Start','Dur','AcqTime','File']

                # Create outfile
                outfile = os.path.join(output,SeriesInstanceUID+'.tsv')

                print("Writing .tsv file with series info: " + outfile)

                with open(outfile, 'w', newline='') as tsvfile:
                    # Create csv writer with tab delimiter
                    writer = csv.writer(tsvfile, delimiter='\t')

                    # Write header row
                    writer.writerow(fieldnames)

                    # Write all rows to file
                    writer.writerows(studydict[StudyInstanceUID][SeriesInstanceUID]['mlist'])
    return studydict

def dcmmlistread(filename: str):
    mlist = []
    with open(filename,'r') as tsvfile:
        # Fieldnames
        fieldnames = ['Plane','Frame','Start','Dur','AcqTime','File']
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            mlist.append([int(row['Plane']), int(row['Frame']), int(row['Start']), int(row['Dur']), row['AcqTime'], row['File']])
    
    return mlist

def suv(data: np.ndarray, dose: int, weight: int):
	""" Calculate Standardizd Uptake Value

	Parameters
	----------
	data : np.ndarray
	   Numpy array in memory containing 4D data

	dose : int
	   radionuclide total dose administered in Bq

	weight : int
	   Patien weight in kg
	"""

	# What about temporal weighting?
	#[g/MBq]
	suv = (np.mean(data, axis=-1) * weight*1000)/dose

	return suv

def count_clusters(mask: np.ndarray):
    _, nclusters = label(mask, return_num=True)

    return nclusters

def threshold_clusters(mask: np.ndarray, volthreshold: float):
    """ Keep only clusters above specified volume threshold
    """
    label_img, nclusters = label(mask, return_num=True)

    if nclusters > 1:
        print(f'Found {nclusters} clusters!')
        print('Keeping only largest cluster')

        # Determine size of each cluster
        clustersize = np.zeros((nclusters,))
        for cluster in range(1,nclusters+1):
            clustersize[cluster-1] = np.count_nonzero(label_img==cluster)
        
        nclusters -= np.count_nonzero(clustersize<=volthreshold)

        # Create a mask with containing only largest cluster
        mask = label_img==np.argmax(clustersize)+1

    return mask, nclusters

def tacwrite(FrameTimesStart: np.ndarray,FrameDuration: np.ndarray,tac:np.ndarray,unit:str,outfile:str,label=None):
	""" Export .tac file format for use with Turku PET center kinetics
	"""

	# Create Header
	header = []
	if label is None:
		label = ['tac1']
		if tac.ndim > 1:
			for i in range(2,np.size(tac,1)+1):
				label += ['tac'+str(i)]
	#header = ['start[seconds]', 'end['+unit+']'] + label
	header.extend(('start[seconds]', 'end['+unit+']', label))

	# Concatenate columns of time and signal
	outarray = np.stack((FrameTimesStart,FrameTimesStart+FrameDuration,tac),axis=1)
	# outarray = np.concatenate((np.stack((FrameTimesStart,FrameTimesStart+FrameDuration),axis=1),tac),axis=1)

	with open(outfile, 'w', encoding='UTF8', newline='') as f:
		writer = csv.writer(f, delimiter='\t')

		# write the header
		writer.writerow(header)

		# write multiple rows
		writer.writerows(outarray)

	f.close()

	return outfile