#include "iftGFLIMGraph.h"
/**
 * \file iftRGBBucketDescriptor.c
 * \brief Create GFLIM graphs using the RGB Bucket Descriptor
 * \note The RGB Bucket Descriptor is part of the SIRS, a evaluation metric used in superpixel segmentations: https://github.com/IsabelaBB/SIRS-superpixels
 * \author Isabela Borlido
 */

/*
spx:
	cores: c1=(0,0,255) c2=(255,255,0)
	dist: (255 + 255 + 255) = 765
	freq: c1=0.95 c2=0.05
	se size = 20, c1=19 c2=1
	a média com RBD é a mesma que sem RBD

QUANDO A MÉDIA É DIFERENTE?
	-> quando uma quantidade menor de cores for selecionada no descritor
	
tamanho da imagem = 400 x 400 = 160000
num. spx = 8000
area / spx = ~20 pixels / spx.

-> se o número de buckets for muito alto, mesmo a frequência das cores relevantes será baixa e, portanto, não será possível distinguir o que é relevante do que é ruído.

*/

typedef struct {
    int num_bins; // each bin has num_buckets buckets
    int num_channels;
    float **buckets; // buckets[num_bins]
} iftColorHistogram;

iftColorHistogram createColorHistogram(int num_bins, int num_channels) {
    iftColorHistogram histogram;
    histogram.num_bins = num_bins;
    histogram.buckets = (float **)malloc(num_bins * sizeof(float*));
    for (int i = 0; i < num_bins; i++) 
        histogram.buckets[i] = (float *)iftAlloc(num_channels, sizeof(float));
    return histogram;
}

void destroyColorHistogram(iftColorHistogram *histogram) {
    if (histogram != NULL) {
        for (int i = 0; i < histogram->num_bins; i++)
            iftFree(histogram->buckets[i]);
        iftFree(histogram->buckets);
    }
}

iftColorHistogram computeSuperpixelDescriptor(int superpixel_ref, 
                                            iftMImage *original_img_rgb, 
                                            iftImage *labeled_img, 
                                            int nbuckets, int nSelBuckets,
                                            int norm_val)
{
    int num_histograms = pow(2, original_img_rgb->m) - 1; // number of color histograms per superpixel
    
    iftColorHistogram ColorHistogram = createColorHistogram(num_histograms * nbuckets, original_img_rgb->m);
    double *V = (double *)calloc(num_histograms * nbuckets, sizeof(double));
    
    // Compute the color histogram for each superpixel
    for(int p = 0; p < original_img_rgb->n; p++)
    {
        int spx_index = labeled_img->val[p] - 1; // get the superpixel index

        if(spx_index != superpixel_ref)
            continue; // skip pixels that are not part of the superpixel we are computing the descriptor for

        float *pixel_color = original_img_rgb->val[p]; // get the pixel color
        
        // Find the maximum color
        float max_color = pixel_color[0];
        for(int c = 1; c < original_img_rgb->m; c++)
        {
            if (pixel_color[c] > max_color)
                max_color = pixel_color[c];
        }
        
        // get histogram index M(p)
        int hist_id = 0; // set 000
        int bit = 1;
        for(int c = 0; c < original_img_rgb->m; c++)
        {
            if (pixel_color[c] == max_color){
                hist_id = hist_id | bit; 
            }
            bit = bit << 1;
        }
        // get bucket based on intensity
        int bin = floor(max_color * nbuckets);
        hist_id--;
        
        V[hist_id * nbuckets + bin]++;

        for(int c = 0; c < original_img_rgb->m; c++)
            ColorHistogram.buckets[hist_id * nbuckets + bin][c] += pixel_color[c]; 
    }

    
    iftDHeap *heap = iftCreateDHeap(num_histograms * nbuckets, V);

    // insert nSelBuckets values in the heap
    int b = 0;
    while(heap->last + 1 < nSelBuckets && b < num_histograms * nbuckets)
    {
        if(V[b] > 0)
            iftInsertDHeap(heap, b); 
        b++;
    }
    
    // maintain in the heap the nSelBuckets buckets with the highest values
    for(; b < num_histograms * nbuckets; b++)
    {
        if(V[b] > 0 && V[heap->node[0]] < V[b]){
            iftRemoveDHeap(heap); // remove the bucket with the lowest value
            iftInsertDHeap(heap, b); // insert the bucket with the highest value
        }
    }

    iftColorHistogram ColorHistogramOut = createColorHistogram(heap->last + 1, original_img_rgb->m);
    
    // extract the nSelBuckets buckets with the highest values
    while(!iftEmptyDHeap(heap))
    {
        int removed = iftRemoveDHeap(heap); // remove the bucket with the highest value
        for(int c = 0; c < original_img_rgb->m; c++)
            ColorHistogramOut.buckets[heap->last+1][c] = ColorHistogram.buckets[removed][c] /= V[removed]; // compute the mean color for the bucket
    }
    iftDestroyDHeap(&heap);
    iftFree(V);
    destroyColorHistogram(&ColorHistogram);
    return ColorHistogramOut;
}

iftMImage *RGBBucketDescriptor(iftMImage *original_img_rgb, iftImage *labeled_img, 
                            int num_superpixels, int nbuckets, int nSelBuckets,
                            int norm_val)
{
    if (num_superpixels < 1)
        iftError("Number of superpixels must be positive", __func__);

    if (nbuckets < 1)
        iftError("Number of buckets must be positive", __func__);

    if (nSelBuckets < 1)
        iftError("Number of selected buckets must be positive", __func__);

    if (original_img_rgb == NULL)
        iftError("Original image is NULL", __func__);
    if (labeled_img == NULL)
        iftError("Labeled image is NULL", __func__);
    
    int num_histograms = pow(2, original_img_rgb->m) - 1; // number of color histograms per superpixel
    
    if(nSelBuckets > nbuckets*num_histograms)
        iftError("Number of selected buckets cannot be greater than the number of bins per histogram", __func__);

    iftColorHistogram *ColorHistogram = (iftColorHistogram*)calloc(num_superpixels, sizeof(iftColorHistogram));
    
#pragma omp parallel for 
    for(int spx = 0; spx < num_superpixels; spx++){
        ColorHistogram[spx] = computeSuperpixelDescriptor(spx, original_img_rgb, labeled_img, nbuckets, nSelBuckets, norm_val);
    }

    iftMImage *reconstructed_img = iftCreateMImage(original_img_rgb->xsize, 
                                            original_img_rgb->ysize, 
                                            original_img_rgb->zsize, 
                                            original_img_rgb->m);

#pragma omp parallel for 
    for(int p = 0; p < original_img_rgb->n; p++)
    {
        int spx_index = labeled_img->val[p] - 1; // get the superpixel index
        if(spx_index < 0) continue; // skip pixels outside the mask

        float *pixel_color = original_img_rgb->val[p]; // get the pixel color
        
        float *min_desc_color = NULL;
        float min_distance = INFINITY;
        iftColorHistogram superpixel_descriptor = ColorHistogram[spx_index];
        for(int b = 0; b < superpixel_descriptor.num_bins; b++)
        {
            if(superpixel_descriptor.buckets[b][0] < 0)
                continue; // skip empty buckets

            float *desc_color = superpixel_descriptor.buckets[b];
            float distance = 0.0;
            for(int c = 0; c < original_img_rgb->m; c++)
                distance += (pixel_color[c] - desc_color[c]);// / (float)norm_val;
            
            if(distance < min_distance)
            {
                min_distance = distance;
                min_desc_color = desc_color; // store the closest color
            }
        }

        if(min_desc_color == NULL)
            iftError("No color descriptor found for the superpixel %d", __func__, spx_index);

        for(int c = 0; c < original_img_rgb->m; c++)
            reconstructed_img->val[p][c] = min_desc_color[c]; // accumulate the color values for the superpixel
    }

    for(int spx = 0; spx < num_superpixels; spx++)
        destroyColorHistogram(&ColorHistogram[spx]);
    iftFree(ColorHistogram);

    return reconstructed_img;
}

void usage()
{
    printf(" Usage: \n");
    printf("iftRGBBucketDescriptor P1 P2 P3 P4 P5 \n");
    printf("P1: images diretory \n");
    printf("P2: superpixel images directory \n");
    printf("P3: number of buckets \n");
    printf("P4: number of selected buckets \n");
    printf("P5: reconstructed images directory (output) \n");
}


int main(int argc, char *argv[]) {
    if (argc != 6)
    {
        usage();
        return 1;
    }

    timer *tstart = iftTic();
    char *images_dir = argv[1];
    char *spx_labels_dir = argv[2];
    int nbuckets = atoi(argv[3]);
    int nSelBuckets = atoi(argv[4]);
    char *output_dir = argv[5];

    if (images_dir == NULL || !iftDirExists(images_dir))
        iftError("Image directory does not exist", __func__);

    if (spx_labels_dir == NULL || !iftDirExists(spx_labels_dir))
        iftError("Superpixel image directory does not exist", __func__);

    if (nbuckets < 1)
        iftError("Number of buckets must be positive", __func__);
    if (nSelBuckets < 1)
        iftError("Number of selected buckets must be positive", __func__);
    
    iftFileSet *fs_imgs = iftLoadFileSetFromDirOrCSV(images_dir, 1, false);
    if (fs_imgs->n < 1)
        iftError("No images found in the directory", __func__);

    if (!iftDirExists(output_dir))
        iftMakeDir(output_dir);

    for (int i = 0; i < fs_imgs->n; i++)
    {
        char *basename = NULL;
        char filename[1024], ext[10];
        timer *tstart = iftTic();

        sprintf(ext, "%s", iftFileExt(fs_imgs->files[i]->path));
        basename = iftFilename(fs_imgs->files[i]->path, ext);

        sprintf(filename, "%s/%s.%s", images_dir, basename, ext);
        iftImage *img = iftReadImageByExt(fs_imgs->files[i]->path);
        sprintf(filename, "%s/%s.pgm", spx_labels_dir, basename);
        iftImage *label_img = iftReadImageByExt(filename);

        int num_superpixels = iftMaximumValue(label_img);
        if (num_superpixels < 1)
            iftError("Number of superpixels must be positive", __func__);

        int norm_val = iftNormalizationValue(iftMaximumValue(img));

        iftMImage *original_img_rgb;
        if (iftIsColorImage(img))
            original_img_rgb = iftImageToMImage(img, RGBNorm_CSPACE);
        else
            original_img_rgb = iftImageToMImage(img, GRAYNorm_CSPACE);
        iftDestroyImage(&img);

        iftMImage *result_mimg = RGBBucketDescriptor(original_img_rgb, label_img, num_superpixels, nbuckets, nSelBuckets, norm_val);
        iftDestroyImage(&label_img);
        iftDestroyMImage(&original_img_rgb);

        iftImage *result = iftCreateColorImage(result_mimg->xsize, result_mimg->ysize, result_mimg->zsize, 
                                                result_mimg->m);
        result->dx = result_mimg->dx;
        result->dy = result_mimg->dy;
        result->dz = result_mimg->dz;

#pragma omp parallel for shared(result_mimg, result)
        for (int p=0; p < result->n; p++) {
            iftColor YCbCr,RGB;
            RGB.val[0] = (int)(result_mimg->val[p][0]*norm_val);
            RGB.val[1] = (int)(result_mimg->val[p][1]*norm_val);
            RGB.val[2] = (int)(result_mimg->val[p][2]*norm_val);
            YCbCr = iftRGBtoYCbCr(RGB, norm_val);
            result->val[p] = YCbCr.val[0];
            result->Cb[p] = YCbCr.val[1];
            result->Cr[p] = YCbCr.val[2];
        }
        
        iftDestroyMImage(&result_mimg);
        
        sprintf(filename, "%s/%s.%s", output_dir, basename, ext);
        iftWriteImageByExt(result, filename);
        iftDestroyImage(&result);
        char *time = iftFormattedTime(iftCompTime(tstart, iftToc()));
        printf("image %s Done ... %s\n", basename, time);
        iftFree(basename);
        iftFree(time);
    }
    iftDestroyFileSet(&fs_imgs);
    char *time = iftFormattedTime(iftCompTime(tstart, iftToc()));
    printf("Done ... %s\n", time);
    iftFree(time);
    return 0;
}