from skimage.measure import label
from tqdm import tqdm
import numpy as np
import csv
from matplotlib.collections import LineCollection

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

def imshow(deck: np.ndarray, cmap='gray_r', vmin: float=None, vmax: float=None, aspect: float=1, overlay: np.ndarray=None, overlaycmap='tab20', alpha: float=0.5, outfile: str=None):
	
	# Create figure
	fig, ax = plt.subplots()
	#im = ax.imshow(im, cmap=_PETRainbowCMAP, vmin=vmin, vmax=vmax, aspect=aspect, interpolation='none')
	im = ax.imshow(deck, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect, interpolation='none')
	
	# Add overlay if present
	if overlay is not None:
		imer = ax.imshow(overlay, cmap=overlaycmap, vmin=0, vmax=np.max(overlay), aspect=aspect, interpolation='none', alpha=alpha)
		fig.colorbar(imer, ax=ax, shrink=0.8, label='')
	else:
		fig.colorbar(im, ax=ax, shrink=0.8, label='')
	
	ax.axis('off')
	
	# Check if outfile
	if outfile:
		plt.savefig(outfile)
	else:
		plt.show()
	plt.close()

def ortoshow(background: np.ndarray, cmap='gray_r', vmin: float=None, vmax: float=None, overlay: np.ndarray=None, midpoint:np.ndarray=None, voxdim: np.ndarray=[1,1,1], mip: bool=False, alpha: float=0.5, outfile: str=None):
  """ Orto outputs mid-sagittal, -coronal and -axial slices into one array
      Assumes overlay has same image dimensions as background
      Assumes data is in "ap rl is" orientation
  """
  x,y,z = background.shape

  if midpoint is None:
    midpoint = [(x-1)//2, (y-1)//2, (z-1)//2]

  # Get midpoint slice of the three orthogonal dimensions
  if not mip:
    sag = np.transpose(background[::-1,midpoint[1],:], (1, 0)) # invert axis to get posterior->anterior
    cor = np.transpose(background[midpoint[0],:,:], (1, 0))
    #ax = background[...,midpoint[2]]
  else:
    # Calculate Maximum Intensity Projection (MIP)
    sag = np.transpose(np.max(background[::-1,:,:],axis=1), (1, 0))
    cor = np.transpose(np.max(background,axis=0), (1, 0))
    #ax = np.max(background, axis=2)

  sz_sag = sag.shape
  sz_cor = cor.shape
  #sz_ax = ax.shape

  # Handle voxdim and the resulting aspect in the figure
  aspect = voxdim[2]/voxdim[1]

  sizes = np.array([sz_sag,sz_cor])

  # Calculate number of rows and cols from sizes of 2D images
  rows = np.max(sizes, axis=0)[0]
  cols = np.sum(sizes, axis=0)[1]

  # Allocate orto array
  orto = np.empty((rows,cols))

  rng_sag = range((rows-sz_sag[0])//2,sz_sag[0]+(rows-sz_sag[0])//2)
  rng_cor = range((rows-sz_cor[0])//2,sz_cor[0]+(rows-sz_cor[0])//2)

  # Insert mid-slices into arr
  orto[rng_sag,0:sz_sag[1]] = sag
  orto[rng_cor,sz_sag[1]:sz_sag[1]+sz_cor[1]] = cor

  # If overlay is present
  if not overlay is None:
    if mip:
      # Calculate Maximum Intensity Projection (MIP)
      sag_ovl = np.transpose(np.max(overlay[::-1,:,:],axis=1), (1, 0))
      cor_ovl = np.transpose(np.max(overlay,axis=0), (1, 0))
    else:
      # Get midpoint slice of the three orthogonal dimensions
      sag_ovl = np.transpose(overlay[::-1,midpoint[0],:], (1, 0)) # invert axis to get posterior->anterior
      cor_ovl = np.transpose(overlay[midpoint[1],:,:], (1, 0))

    orto_ovl = np.empty((rows,cols))

    # Insert mid-slices into arr
    orto_ovl[rng_sag,0:sz_sag[1]] = sag_ovl
    orto_ovl[rng_cor,sz_sag[1]:sz_sag[1]+sz_cor[1]] = cor_ovl

    # Create figure
    orto_ovl_masked = np.ma.masked_where(orto_ovl == 0, orto_ovl)
    imshow(orto, overlay=orto_ovl_masked, overlaycmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect, alpha=alpha, outfile=outfile)
  else:
    imshow(orto, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect, outfile=outfile)

  return orto

def montage(I: np.ndarray, arraysize: np.ndarray=None):
	"""
	Generates a single montage array and plots the result using the
	supplied colormap. I is a multimensional array with sides of length (m,n,count).

	Parameters
	----------
	data : np.ndarray
	   Numpy array in memory containing 3D data

	"""

	m,n,count = np.shape(I)

	# Determine best square for montage if not specified by user
	if arraysize is None:
		# Find the best square montage
		c = count/np.arange(1,count+1)
		c = c[np.equal(np.mod(c, 1), 0)]
		r = count/c
		q = (c*n)/(r*m)
		idx = find_nearest(q,1)
		aspect = q[idx]

		if aspect>0.5 and aspect<2:
			nc=int(c[idx])
			nr=int(r[idx])
		else:
			nr=np.arange(1,int(np.ceil(np.sqrt(count))+1))
			nc=np.ceil(count/nr)
			q=(nc*n)/(nr*m)
			idx=np.argwhere(q<1)
			qi=q
			qi[idx] = 1/qi[idx]

			idx=np.argmin(qi)
			nc=int(nc[idx])
			nr=int(nr[idx])
			aspect=q[idx]
	else:
		nr,nc = arraysize

	# Allocate montage
	M = np.zeros((nr * m, nc * n))

	image_id = 0
	for j in range(nr):
		for k in range(nc):
			if image_id >= count:
				break
			sliceM, sliceN = j * m, k * n
			# M[sliceN:sliceN + n, sliceM:sliceM + m] = I[:, :, image_id]
			M[sliceM:sliceM + m, sliceN:sliceN + n] = I[:, :, image_id]
			image_id += 1

	return M

def plot_outlines(bool_img, ax=None, **kwargs):
  if ax is None:
    ax = plt.gca()

  edges = get_all_edges(bool_img=bool_img)
  edges = edges - 0.5  # convert indices to coordinates; TODO adjust according to image extent
  outlines = close_loop_edges(edges=edges)
  cl = LineCollection(outlines, **kwargs)
  ax.add_collection(cl)

def get_all_edges(bool_img):
  """
  Get a list of all edges (where the value changes from True to False) in the 2D boolean image.
  The returned array edges has he dimension (n, 2, 2).
  Edge i connects the pixels edges[i, 0, :] and edges[i, 1, :].
  Note that the indices of a pixel also denote the coordinates of its lower left corner.
  """
  edges = []
  ii, jj = np.nonzero(bool_img)
  for i, j in zip(ii, jj):
    # North
    if j == bool_img.shape[1]-1 or not bool_img[i, j+1]:
      edges.append(np.array([[i, j+1],
                                  [i+1, j+1]]))
    # East
    if i == bool_img.shape[0]-1 or not bool_img[i+1, j]:
      edges.append(np.array([[i+1, j],
                                  [i+1, j+1]]))
    # South
    if j == 0 or not bool_img[i, j-1]:
      edges.append(np.array([[i, j],
                                  [i+1, j]]))
    # West
    if i == 0 or not bool_img[i-1, j]:
      edges.append(np.array([[i, j],
                                  [i, j+1]]))

  if not edges:
    return np.zeros((0, 2, 2))
  else:
    return np.array(edges)


def close_loop_edges(edges):
  """
  Combine thee edges defined by 'get_all_edges' to closed loops around objects.
  If there are multiple disconnected objects a list of closed loops is returned.
  Note that it's expected that all the edges are part of exactly one loop (but not necessarily the same one).
  """

  loop_list = []
  while edges.size != 0:

    loop = [edges[0, 0], edges[0, 1]]  # Start with first edge
    edges = np.delete(edges, 0, axis=0)

    while edges.size != 0:
      # Get next edge (=edge with common node)
      ij = np.nonzero((edges == loop[-1]).all(axis=2))
      if ij[0].size > 0:
        i = ij[0][0]
        j = ij[1][0]
      else:
        loop.append(loop[0])
        # Uncomment to to make the start of the loop invisible when plotting
        # loop.append(loop[1])
        break

      loop.append(edges[i, (j + 1) % 2, :])
      edges = np.delete(edges, i, axis=0)

    loop_list.append(np.array(loop))

  return loop_list