import numpy as np
import nibabel as nib
import json
import argparse
import os
from os import makedirs
from os.path import isdir, splitext
import matplotlib.pyplot as plt
from skimage.measure import label

import lib

__scriptname__ = 'idif'
__version__ = '0.3.1'

def aorta_segment(aortamask: np.ndarray=bool):
  """ Segment aorta in four segments with value:
    1. Aorta Ascendens
    2. Aorta Arch
    3. Aorta Descendens (upper)
    4. Aorta Descendens (lower)
  """

  # Get image dimensions
  xdim,ydim,nslices = aortamask.shape

  # Allocate aortamask_segmented
  aortamask_segmented = aortamask.astype(int)

  # Loop over all axial slices and count number of clusters
  nclusters = np.zeros((nslices,),dtype=int)

  for slc in range(nslices):
      label_img, nclusters[slc] = label(aortamask[:,:,slc], return_num=True)

  # Compute volume within each slice
  volume = np.count_nonzero(aortamask,axis=(0,1))
  print(np.mean(volume[nclusters==1]),np.mean(volume[nclusters==2]))

  # Correct
  nclusters[nclusters>2] = 2

  # Get start, stop index pairs for islands/seq. of 1s
  idx_pairs = np.where(np.diff(np.hstack(([False],nclusters==1,[False]))))[0].reshape(-1,2)

  # Get the island lengths, whose argmax would give us the ID of longest island.
  # Start index of that island would be the desired output
  start_longest_seq_1 = idx_pairs[np.diff(idx_pairs,axis=1).argmax(),0]

  # Aorta descendens is the biggest island of connected slices
  slices_desc = range(start_longest_seq_1,nslices)

  # Segment aorta descendens lower
  aortamask_segmented[:,:,slices_desc] = aortamask[:,:,slices_desc] * 4

  # Aorta arch is the biggest island of connected twos
  nclusters_twos = nclusters
  nclusters_twos[nclusters_twos==1] = 0

  # Get start, stop index pairs for islands/seq. of 1s
  idx_pairs = np.where(np.diff(np.hstack(([False],nclusters==2,[False]))))[0].reshape(-1,2)
  # Get the island lengths, whose argmax would give us the ID of longest island.
  # Start index of that island would be the desired output
  start_longest_seq_2 = idx_pairs[np.diff(idx_pairs,axis=1).argmax(),0]
  slices_arch = range(0,start_longest_seq_2)
  slices_two = range(start_longest_seq_2,start_longest_seq_1)

  # Find upper descending part of aorta by finding connection with lower part
  label_img_2, nclusters = label(aortamask[:,:,slices_two[0]:slices_desc[0]+1], return_num=True)
  for cluster in range(1,nclusters+1):
    clustersize = np.sum((label_img_2==cluster))
    if np.sum((label_img_2==cluster)*(aortamask_segmented[:,:,slices_two[0]:slices_desc[0]+1]==4)):
      aortamask_segmented[:,:,slices_two[0]:slices_desc[0]] =  aortamask_segmented[:,:,slices_two[0]:slices_desc[0]] + (label_img_2[:,:,0:-1]==cluster)*2

    aortamask_segmented[:,:,slices_arch] = aortamask[:,:,slices_arch] * 2
  return aortamask_segmented

#def main():
if __name__ == '__main__':
  # Create argument parser
  parser = argparse.ArgumentParser(prog='IDIF', description='Image derived input function of dynamic PET data')
  parser.add_argument('-v','--version', action='version', version='%(prog)s {version}'.format(version=__version__))

  # Required arguments
  parser.add_argument('-i','--data', help='Input directory (4D Nifti)', required=True)
  parser.add_argument('-s','--segmentation', help='Organ segmentation directory (DICOM)', required=True)
  parser.add_argument('-l','--labelmap', help='Labelmap type', choices=['TotalSegmentator','custom'], default='TotalSegmentator', required=False)
  parser.add_argument('-a','--aortaidx', help='Aorta idx in the segmentation (if custom labelmap)', default=52, type=int, required=False)
  parser.add_argument('-o','--outdir', help='Output directory', required=True)

  # Parse arguments
  args = parser.parse_args()

  PETDATAPATH = args.data
  LABELPATH = args.segmentation
  LABELMAPTYPE = args.labelmap
  AORTAIDX = args.aortaidx

  if not isdir(args.outdir):
    makedirs(args.outdir)

  # Read segmentation data into deck
  labelObj = nib.load(LABELPATH)
  hdr_label = labelObj.header
  deck_label = labelObj.get_fdata()

  xdim,ydim,nslices = deck_label.shape
      
  print(f"Label dimensions: {deck_label.shape}")

  # Get label value for aorta in segmentation
  if LABELMAPTYPE == 'TotalSegmentator':
    regionidx = lib.get_regionidx('aorta') # totalsegmentator-v2
  elif LABELMAPTYPE == 'custom':
    regionidx = AORTAIDX # custom
  else:
    raise SystemExit('No label match')

  print(f'Method: {LABELMAPTYPE}, Aorta label: {regionidx}')

  # Get only aorta from segmentation mask
  aortamask = (deck_label == regionidx)

  # Get slices containing region
  xmin, xsize, ymin, ysize, zmin, zsize = lib.bbox(aortamask)
  # Get slices containing region
  slicesum = np.sum(aortamask.astype(int), axis=(0, 1))
  slices_nonzero = [i for i, e in enumerate(slicesum) if e > 0]
  aortamask = aortamask[:,:,zmin:zmin+zsize]

  ### Read Dynamical Data (Only slices within aorta) ###
  # Create mlist dictionary of all files in PET directory
  print("Load PET data:")
  dataObj = nib.load(PETDATAPATH)
  hdr_pet = dataObj.header
  deck = dataObj.get_fdata()
  deck = deck[:,:,slices_nonzero,:]

  # Get Spatial information
  xdim,ydim,nslices,nframes = deck.shape
  #xdim,ydim,nslices = deck.shape
  print(PETDATAPATH)
  if '.nii.gz' in PETDATAPATH:
    JSONPATH = splitext(splitext(PETDATAPATH)[0])[0]
  else:
    JSONPATH = splitext(PETDATAPATH)[0]
  try:
    with open(JSONPATH + '.json','r') as fr:
      PETMETADATA = json.load(fr)
  except FileNotFoundError as fnfe:
    raise SystemExit('Did not find the needed .json file with frametimes from dcm2niix')
  print(PETMETADATA['FrameTimesStart'])
  print(hdr_pet['pixdim'][1],hdr_pet['pixdim'][2])
  #exit()
  voxdim = np.array([hdr_pet['pixdim'][1], hdr_pet['pixdim'][2], hdr_pet['pixdim'][3]])
    
  # Get time info from header
  FrameTimesStart = np.array(PETMETADATA['FrameTimesStart'])
  FrameDuration = np.array(PETMETADATA['FrameDuration'])
  MidFrameTime = FrameTimesStart + FrameDuration/2.0

  # Create SUV from first 40 second frames
  frames = FrameDuration == np.unique(FrameDuration)[0]
  #meanframes = FrameTimesStart < 40
  mean = np.mean(deck[...,frames], axis=-1)

  # Compute median SUV value inside aortamask
  median = np.median(mean[aortamask])
  print(f'Median SUV inside Aorta Mask: {median}')
  #exit()
  
  # Create SUV from 5 to 40 seconds
  #print(f"Create SUV map. TotalDose: {hdr_pet['RadionuclideInjectedDose']/1e6} MBq, PatientWeight: {hdr_pet['PatientWeight']} kg")
  # Create SUV from first 40 second frames
  #suvframes = FrameDuration==np.unique(FrameDuration)[0]
  #suvframes = FrameTimesStart < 40

  #SUV = lib.suv(deck[:,:,:,suvframes],hdr_pet['RadionuclideInjectedDose'],hdr_pet['PatientWeight'])

  # Compute median SUV value inside aortamask
  #SUV_median = np.median(SUV[aortamask])
  #print(f'Median SUV inside Aorta Mask: {SUV_median}')

  # Create figure of SUV overlayed with aorta VOI
  #fig, ax = plt.subplots(1,2, constrained_layout=True)
  # Sag
  #ax[0].imshow(np.transpose(np.max(SUV[::-1,:,:],axis=1), (1, 0)), vmin=0, vmax=2*SUV_median, cmap='gray_r', 
  #    aspect=hdr_pet['SliceThickness']/hdr_pet['PixelSpacing'][1])
  #plotting.plot_outlines(np.transpose(np.max(aortamask[::-1,:,:],axis=1), (1, 0)).T, ax=ax[0], lw=0.5, color='r')
  #ax[0].axis('off')
  # Cor
  #ax[1].imshow(np.transpose(np.max(SUV,axis=0), (1, 0)), vmin=0, vmax=SUV_median*2, cmap='gray_r', 
  #    aspect=hdr_label['SliceThickness']/hdr_label['PixelSpacing'][0])
  #plotting.plot_outlines(np.transpose(np.max(aortamask,axis=0), (1, 0)).T, ax=ax[1], lw=0.5, color='r')
  #ax[1].axis('off')
  #plt.suptitle('Aorta Segmentation')
  #plt.savefig(os.path.join(args.outdir,'segmentation.pdf'))

  # Threshold aortamask with median(SUV)/1.5
  aortamask = aortamask*np.int8(mean>median/1.5)

  # Count number of clusters
  nclusters = lib.count_clusters(aortamask)
  print(f'Number of Clusters found in Aorta Segmentation: {nclusters}')

  if nclusters > 1:
    # Handle the mystery
    print('Handling multiple clusters')

    # Keep only cluster above threshold
    volthreshold=20
    print(f'Removing cluster(s) with volume lower than {volthreshold*np.prod(voxdim)/1000:.2f} ml')
    aortamask_tmp, nclusters = lib.threshold_clusters(aortamask, volthreshold=volthreshold)

    print(f'  Remaining clusters: {nclusters}')

    # Still have multiple clusters - now try to extrapolate
    if nclusters > 1:
      print('Extrapolation')

      # Loop over axial slices and count number of clusters
      nrois = np.zeros((nslices,),dtype=int)
      for slc in range(nslices-1,-1,-1):
        nrois[slc] = roi.count_clusters(aortamask[:,:,slc])

        if nrois[slc] == 0 or np.count_nonzero(aortamask[:,:,slc])<3:
          # Get bounding box for the two slices below
          xmin_tmp, xsize_tmp, ymin_tmp, ysize_tmp, _, _ = roi.bbox(aortamask[:,:,slc+1:slc+3])

          xmid_tmp = xmin_tmp+xsize_tmp//2
          ymid_tmp = ymin_tmp+ysize_tmp//2

          # Keep only largest cluster if multiple
          maskimg = roi.keep_largest_cluster(SUV[xmid_tmp-5:xmid_tmp+5,ymid_tmp-5:ymid_tmp+5,slc]>SUV_median/1.5)

          # Run region props on binary mask add SUV image for weighted centroid estimation
          regions = regionprops(label(maskimg), intensity_image=SUV[xmid_tmp-5:xmid_tmp+5,ymid_tmp-5:ymid_tmp+5,slc])
          for props in regions:
            aortamask[xmid_tmp-5+int(props.centroid_weighted[0])-3:xmid_tmp-5+int(props.centroid_weighted[0])+4,
            ymid_tmp-5+int(props.centroid_weighted[1])-3:ymid_tmp-5+int(props.centroid_weighted[1])+4,slc] = 1
              
        # Dilate and threshold to account for aortavoxels outside segmentation
        aortamask_dilated = binary_dilation(aortamask[:,:,slc])
        aortamask[:,:,slc] = aortamask_dilated*np.int8(SUV[:,:,slc]>SUV_median/1.5)

  # Create figure of SUV overlayed with aorta VOI
  fig, ax = plt.subplots(1,2, constrained_layout=True)
  # Sag
  ax[0].imshow(np.transpose(np.max(mean[::-1,:,:],axis=1), (1, 0)), vmin=0, vmax=2*median, cmap='gray_r', 
      aspect=voxdim[-1]/voxdim[0])
  lib.plot_outlines(np.transpose(np.max(aortamask[::-1,:,:],axis=1), (1, 0)).T, ax=ax[0], lw=0.5, color='r')
  ax[0].axis('off')
  # Cor
  ax[1].imshow(np.transpose(np.max(mean,axis=0), (1, 0)), vmin=0, vmax=median*2, cmap='gray_r', 
      aspect=voxdim[-1]/voxdim[0])
  lib.plot_outlines(np.transpose(np.max(aortamask,axis=0), (1, 0)).T, ax=ax[1], lw=0.5, color='r')
  ax[1].axis('off')
  plt.suptitle('Aorta Segmentation')
  plt.savefig(os.path.join(args.outdir,'segmentation_corr.pdf'))
  print(os.path.join(args.outdir,'segmentation_corr.pdf'))

  ### Segment aorta in four segments ###
  segments = ['Aorta asc', 'Aortic arch', 'Aorta desc (upper)', 'Aorta desc (lower)']
  aortamask_segmented = aorta_segment(aortamask)

  # Create bounding box for figure
  xmin, xsize, ymin, ysize, zmin, zsize = roi.bbox(aortamask_segmented>0)

  # Create square box with original centers
  xmid = xmin+xsize//2
  ymid = ymin+ysize//2
  xsize = ysize = np.amax([xsize, ysize])
  xmin = xmid-xsize//2
  ymin = ymid-ysize//2

  M = lib.montage(aortamask_segmented[xmin:xmin+xsize,ymin:ymin+ysize,zmin:zmin+zsize])
  lib.imshow(M,vmin=0,vmax=4,cmap='viridis',outfile=os.path.join(args.outdir,'aorta.pdf'))

  # Create plot for evalutating aorta mask position on SUV PET
  plotting.ortoshow(SUV,overlay=aortamask_segmented, cmap='tab20', vmin=0, vmax=20, voxdim=voxdim, mip=True, outfile=os.path.join(args.outdir,'mask_orto.pdf'))

  ### Create VOI inside aorta arch of approx 1 mL ###
  thr = 1000//np.prod(voxdim)

  # Allocate
  VOI = np.zeros((xdim,ydim,nslices,4), dtype=bool)
  idif = np.zeros((nframes,4))

  print('Looping over each segment')
  N = int(np.round((1000/(np.prod(voxdim)*3*3))/2)*2)
  print(f"Length of VOI: {N} slices")
  for seg in range(4):
    if seg in [0,2,3]:
      # Create slice profile in z-direction
      # Get median value of each slice
      slicemedian = np.zeros(nslices)
      for slc in range(nslices):
        if np.sum(aortamask_segmented[:,:,slc]==seg+1):
          SUV_segment = SUV[:,:,slc]*(aortamask_segmented[:,:,slc]==seg+1)
          slicemedian[slc] = np.median(SUV_segment[SUV_segment>0])

      # Sliding window average of slice profile
      middleslc = np.argmax(np.convolve(slicemedian, np.ones(N)/N, mode='valid'))+N//2

      # Position 3x3xN VOI within segment with detected center slice
      for slc in range(middleslc-N//2,middleslc+N//2):
        if np.sum(aortamask_segmented[:,:,slc]==seg+1):
          x0,y0 = roi.cog(SUV[:,:,slc]*(aortamask_segmented[:,:,slc]==seg+1))
          VOI[y0-1:y0+2,x0-1:x0+2,slc,seg] = 1
    else:
      # Aortic Arch
      # Data-driven approach to find VOI based on maximum thresholding
      prc = 99.99
      volume = 0
        
      while volume <= thr:
        VOI[:,:,:,seg] = (SUV*(aortamask_segmented==seg+1))>=np.percentile(SUV[aortamask_segmented==seg+1],prc)
        label_img, nclusters = label(VOI[...,seg], return_num=True)
          
        clustersize = np.zeros((nclusters,))
            
        for cluster in range(nclusters):
          clustersize[cluster] = np.sum(label_img==cluster+1)
            
        # Find largest cluster
        maxclusteridx = np.argmax(clustersize)+1
        VOI[:,:,:,seg] = label_img==maxclusteridx
            
        volume = clustersize[maxclusteridx-1]
        prc -= 0.5
        
      print(f"volume: {volume*np.prod(voxdim):.2f} mm3")
      print(f"threshold: {prc}")

    ### Extract IDIF as the mean inside the VOI ###
    idif[:,seg] = np.mean(deck[np.squeeze(VOI[:,:,:,seg])], axis=0)

    ### Write NIfTI file of SUV data
    #niftifile = os.path.join(args.outdir,'VOI_segment-'+str(seg+1)+'.nii.gz')
    #if not os.path.isfile(niftifile):
    #    print('Store VOI as NIfTI')
    #    # Use transformation matrix from DICOM header
    #    OM = hdr_pet['Affine3D']

    #    # Create NIFTI object
    #    header = nib.Nifti1Header()
    #    header.set_data_shape(VOI[:,:,:,seg].shape)
    #    header.set_dim_info(slice=2)
    #   header.set_xyzt_units('mm')

    #    ### TODO: LIKELY MORE INFO SHOULD BE ADDED TO THE NIFTI FILE HERE ###
    #    nifti =  nib.Nifti1Image(VOI[:,:,:,seg], OM, header=header)
    #    nib.save(nifti, niftifile)

    # Save VOI as numpy array
    voifile = os.path.join(args.outdir,'VOI.npy')
    #np.save(voifile,VOI)

    # Write IDIF to file
    lib.tacwrite(FrameTimesStart,FrameDuration,idif[:,seg],'Bq/cc',os.path.join(args.outdir,'IDIF_'+method.lower()+'_segment-'+str(seg+1)+'.tac'),['idif'])

    # Create plot for evalutating aorta mask position on SUV PET
    plotting.ortoshow(mean,overlay=VOI[...,0]+(VOI[...,1]*2)+(VOI[...,2]*3)+(VOI[...,3]*4), vmin=0, vmax=20, cmap='tab20', voxdim=voxdim, mip=True, outfile=os.path.join(args.outdir,'VOI_orto.pdf'))
    print(os.path.join(args.outdir,'VOI_orto.pdf'))

    # Save a tsv file with IDIF stats
    tsvfilename = os.path.join(args.outdir,method.lower()+'_IDIF.tsv')
    print('Finished')
    exit()
    # Headers
    fieldnames = ['Method','Radiopharmaceutical','PatientWeight','RadionuclideTotalDose','Segment','Region','Unit','AUC40','Peak','Slope']
  with open(tsvfilename,'w') as tsvfile:
    # Create csv writer with tab delimiter
    writer = csv.writer(tsvfile, delimiter='\t')

    # Write header row
    writer.writerow(fieldnames)

    for seg in range(4):
      # Calculate metrics in native units (typical Bq/ml)
      auc_bq = np.trapz(idif[suvframes,seg], x=MidFrameTime[suvframes])
      peak_bq = np.amax(idif[:,seg])# Peak value is found within the entire duration
      slope_bq = np.amax(np.diff(idif[suvframes,seg]))

      writer.writerow([method,hdr_pet['Radiopharmaceutical'],hdr_pet['PatientWeight'],hdr_pet['RadionuclideTotalDose'],seg+1,segments[seg],hdr_pet['Unit'],auc_bq,peak_bq,slope_bq])

      # Calculate metrics in SUV units
      idif_suv = idif[:,seg] * hdr_pet['PatientWeight'] * 1000 / hdr_pet['RadionuclideTotalDose']
      auc_suv = np.trapz(idif_suv[suvframes], x=MidFrameTime[suvframes])
      peak_suv = np.amax(idif_suv) # Peak value is found within the entire duration
      slope_suv = np.amax(np.diff(idif_suv[suvframes]))

      writer.writerow([method,hdr_pet['Radiopharmaceutical'],hdr_pet['PatientWeight'],hdr_pet['RadionuclideTotalDose'],seg+1,segments[seg],'SUV',auc_suv,peak_suv,slope_suv])


  # Create figure
  fig, ax = plt.subplots(2,1)
  ax[0].plot(MidFrameTime, idif[:,0], label=segments[0])
  ax[0].plot(MidFrameTime, idif[:,1], label=segments[1])
  ax[0].plot(MidFrameTime, idif[:,2], label=segments[2])
  ax[0].plot(MidFrameTime, idif[:,3], label=segments[3])
  ax[0].legend()
  ax[0].set_ylabel('Concentration ' + hdr_pet['Unit'])

  ax[1].plot(MidFrameTime[suvframes], idif[suvframes,0])
  ax[1].plot(MidFrameTime[suvframes], idif[suvframes,1])
  ax[1].plot(MidFrameTime[suvframes], idif[suvframes,2])
  ax[1].plot(MidFrameTime[suvframes], idif[suvframes,3])
  ax[1].set_xlabel('Time [s]')
  ax[1].set_ylabel('Concentration ' + hdr_pet['Unit'])
  plt.savefig(os.path.join(args.outdir,method+'_IDIF.pdf'), format='pdf')
  plt.close()