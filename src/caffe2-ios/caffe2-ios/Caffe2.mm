//
//  Caffe2.m
//  caffe2-ios
//
//  Created by Kaiwen Yuan on 2017-04-28.
//
//

#import <Foundation/Foundation.h>
#import "Caffe2.h"
#include "caffe2/core/predictor.h"
#include "caffe2/utils/proto_utils.h"


void ReadProtoIntoNet(std::string fname, caffe2::NetDef* net) {
    int file = open(fname.c_str(), O_RDONLY);
    CAFFE_ENFORCE(net->ParseFromFileDescriptor(file));
    close(file);
}

CGContextRef CreateRGBABitmapContext (CGImageRef inImage)
{
    CGContextRef    context = NULL;
    CGColorSpaceRef colorSpace;
    void *          bitmapData;
    int             bitmapByteCount;
    int             bitmapBytesPerRow;
    
    // Get image width, height. We'll use the entire image.
    size_t pixelsWide = CGImageGetWidth(inImage);
    size_t pixelsHigh = CGImageGetHeight(inImage);
    
    // Declare the number of bytes per row. Each pixel in the bitmap in this
    // example is represented by 4 bytes; 8 bits each of red, green, blue, and
    // alpha.
    bitmapBytesPerRow   = int(pixelsWide * 4);
    bitmapByteCount     = int(bitmapBytesPerRow * pixelsHigh);
    
    // Use the generic RGB color space.
    colorSpace = CGColorSpaceCreateWithName(kCGColorSpaceGenericRGB);
    if (colorSpace == NULL)
    {
        fprintf(stderr, "Error allocating color space\n");
        return NULL;
    }
    
    // Allocate memory for image data. This is the destination in memory
    // where any drawing to the bitmap context will be rendered.
    bitmapData = malloc( bitmapByteCount );
    if (bitmapData == NULL)
    {
        fprintf (stderr, "Memory not allocated!");
        CGColorSpaceRelease( colorSpace );
        return NULL;
    }
    
    // Create the bitmap context. We want pre-multiplied ARGB, 8-bits
    // per component. Regardless of what the source image format is
    // (CMYK, Grayscale, and so on) it will be converted over to the format
    // specified here by CGBitmapContextCreate.
    context = CGBitmapContextCreate (bitmapData,
                                     pixelsWide,
                                     pixelsHigh,
                                     8,      // bits per component
                                     bitmapBytesPerRow,
                                     colorSpace,
                                     kCGImageAlphaPremultipliedLast);
    if (context == NULL)
    {
        free (bitmapData);
        fprintf (stderr, "Context not created!");
    }
    
    // Make sure and release colorspace before returning
    CGColorSpaceRelease( colorSpace );
    
    return context;
}

@interface Caffe2(){
    caffe2::NetDef _initNet;
    caffe2::NetDef _predictNet;
    caffe2::Predictor *_predictor;
}

@property (atomic, assign) BOOL busyWithInference;

@property (weak, nonatomic) IBOutlet UIImageView *originalImageView;
@property (weak, nonatomic) IBOutlet UIImageView *finalImageView;
//@property (nonatomic) NSURL *imageURL;
@property (weak, nonatomic) IBOutlet UILabel *resultLabel;
@property (nonatomic) CVPixelBufferRef *pixelRef;


@end

@implementation Caffe2

- (UIImage *)stepOne: (UIImage*) originalImage {
    
    // NSURL* imageURL = [NSURL URLWithString:stringURL];

    // NSData *data = [NSData dataWithContentsOfURL:imageURL];
    // UIImage *originalImage = [UIImage imageWithData:data];
    CGSize newSize = CGSizeMake(256, 256);
    UIGraphicsBeginImageContextWithOptions(newSize, YES, 1.0);
    [originalImage drawInRect:CGRectMake(0, 0, newSize.width, newSize.height)];
    
    UIImage *newImage = UIGraphicsGetImageFromCurrentImageContext();
    UIGraphicsEndImageContext();
    return newImage;
}



- (UIImage *) stepTwo:(UIImage *) originalImage width:(CGFloat)width height:(CGFloat) height {
    CGFloat y = (originalImage.size.height - height)/2;

    CGFloat x = (originalImage.size.width - width)/2;
    
    CGRect rect = CGRectMake(x, y, width, height);
    CGImageRef imageRef = CGImageCreateWithImageInRect(originalImage.CGImage, rect);
    UIImage *newImage = [UIImage imageWithCGImage: imageRef scale: originalImage.scale orientation: originalImage.imageOrientation];

    return newImage;
}


- (nullable CVPixelBufferRef) pixelBufferFromImage2: (UIImage*) image {

    NSDictionary *attributes = @{
                                  (__bridge NSString *)kCVPixelBufferCGImageCompatibilityKey: @(YES),
                                  (__bridge NSString *)kCVPixelBufferCGBitmapContextCompatibilityKey: @(YES)};
    
    CVPixelBufferRef pixelBuffer;
    CVReturn status = CVPixelBufferCreate(kCFAllocatorDefault, (int) image.size.width, (int) image.size.height, kCVPixelFormatType_32ARGB, (__bridge CFDictionaryRef) attributes, &pixelBuffer);
    
    if (status != kCVReturnSuccess) {
        return nil;
    }
    
    CVPixelBufferLockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    void (*data) = CVPixelBufferGetBaseAddress(pixelBuffer);
    CGColorSpaceRef rgbColorSpace = CGColorSpaceCreateDeviceRGB();
    struct CGContext *context = CGBitmapContextCreate(data,(int) image.size.width, (int) image.size.height, 8, CVPixelBufferGetBytesPerRow(pixelBuffer), rgbColorSpace, kCGImageAlphaNoneSkipFirst);
    CGContextTranslateCTM(context, 0, image.size.height);
    CGContextScaleCTM(context, 1.0, -1.0);
    UIGraphicsPushContext(context);
    [image drawInRect:CGRectMake(0, 0, (int) image.size.width, (int) image.size.height)];
    UIGraphicsPopContext();
    CVPixelBufferUnlockBaseAddress(pixelBuffer, kCVPixelBufferLock_ReadOnly);
    
    return pixelBuffer;

}

- (UIImage *) convertBacktoImage: (CVPixelBufferRef) pixelBuffer {
    
    CIImage *ciImage = [CIImage imageWithCVPixelBuffer:pixelBuffer];

    CIContext *temporaryContext = [CIContext contextWithOptions:nil];
    CGImageRef videoImage = [temporaryContext
                             createCGImage:ciImage
                             fromRect:CGRectMake(0, 0,
                                                 CVPixelBufferGetWidth(pixelBuffer),
                                                 CVPixelBufferGetHeight(pixelBuffer))];

    UIImage *uiImage = [UIImage imageWithCGImage:videoImage];
    CGImageRelease(videoImage);

    return uiImage;
 
}

- (NSString*)pathToResourceNamed:(NSString*)name error:(NSError **)error {
    NSString* netName = [[NSBundle mainBundle] pathForResource:name ofType: @"pb"];
    if (netName == NULL) {
        NSMutableDictionary* details = [NSMutableDictionary dictionary];
        [details setValue:[NSString stringWithFormat:@"File named \"%@\" not found in main bundle", name] forKey:NSLocalizedDescriptionKey];
        NSLog(@"File named \"%@\" not found in main bundle", name);
        *error = [[NSError alloc] initWithDomain:@"Caffe2" code:1 userInfo:details];
        return nil;
    }
    return netName;
}

- (instancetype) init:(nonnull NSString*)initNetFilename predict:(nonnull NSString*)predictNetFilename error:(NSError **)error {
    self = [super init];
    if(self){
        NSString* initNetPath = [self pathToResourceNamed:initNetFilename error:error];
        NSString* predictNetPath = [self pathToResourceNamed:predictNetFilename error:error];
        
        if (initNetPath == nil || predictNetPath == nil) {
            return nil;
        }
        ReadProtoIntoNet(initNetPath.UTF8String, &_initNet);
        ReadProtoIntoNet(predictNetPath.UTF8String, &_predictNet);
        
        _predictNet.set_name("PredictNet");
        _predictor = new caffe2::Predictor(_initNet, _predictNet);
    }
    return self;
}

-(void)dealloc {
    google::protobuf::ShutdownProtobufLibrary();
}

- (nullable NSArray<NSNumber*>*) predict:(nonnull UIImage*) start_image{
    NSMutableArray* result = nil;
    caffe2::Predictor::TensorVector output_vec;
    
    if (self.busyWithInference) {
        return nil;
    } else {
        self.busyWithInference = true;
    }
    UIImage *resized = [self stepOne:start_image];
    UIImage *image224 = [self stepTwo:resized width:224 height:224];
    CVPixelBufferRef bufferRef = [self pixelBufferFromImage2:image224];
    NSLog(@"Buff = %@", bufferRef);
    UIImage *image = [self convertBacktoImage:bufferRef];

    CGImageRef inImage = image.CGImage;
    // Create the bitmap context
    // We do this to ensure correct color space layout
    CGContextRef cgctx = CreateRGBABitmapContext(inImage);
    if (cgctx == NULL){
        return nil;
    }
    
    // Get image width, height. We'll use the entire image.
    size_t w = CGImageGetWidth(inImage);
    size_t h = CGImageGetHeight(inImage);
    NSLog(@"h = %zu", h);
    NSLog(@"w = %zu", w);
    CGRect rect = {{0,0},{static_cast<CGFloat>(w),static_cast<CGFloat>(h)}};
    
    // Draw the image to the bitmap context. Once we draw, the memory
    // allocated for the context for rendering will then contain the
    // raw image data in the specified color space.
    CGContextDrawImage(cgctx, rect, inImage);
    void *data = CGBitmapContextGetData (cgctx);
    if (_predictor && data) {
        UInt8* pixels = (UInt8*) data;
        caffe2::TensorCPU input;
        
        // Reasonable dimensions to feed the predictor.
        const int predHeight = (int)CGSizeEqualToSize(self.imageInputDimensions, CGSizeZero) ? int(h) : self.imageInputDimensions.height;
        const int predWidth = (int)CGSizeEqualToSize(self.imageInputDimensions, CGSizeZero) ? int(w) : self.imageInputDimensions.width;
        const int crops = 1;
        const int channels = 3;
        const int size = predHeight * predWidth;
        const float hscale = ((float)h) / predHeight;
        const float wscale = ((float)w) / predWidth;
        const float scale = std::min(hscale, wscale);
        std::vector<float> inputPlanar(crops * channels * predHeight * predWidth);
        // Scale down the input to a reasonable predictor size.
        for (auto i = 0; i < predHeight; ++i) {
            const int _i = (int) (scale * i);
            for (auto j = 0; j < predWidth; ++j) {
                const int _j = (int) (scale * j);
                // The input is of the form RGBA, we only need the RGB part.
                float red = (float) pixels[(_i * w + _j) * 4 + 0];
                float green = (float) pixels[(_i * w + _j) * 4 + 1];
                float blue = (float) pixels[(_i * w + _j) * 4 + 2];
                
                inputPlanar[i * predWidth + j + 0 * size] = blue;
                inputPlanar[i * predWidth + j + 1 * size] = green;
                inputPlanar[i * predWidth + j + 2 * size] = red;
            }
        }
        
        input.Resize(std::vector<int>({crops, channels, predHeight, predWidth}));
        input.ShareExternalPointer(inputPlanar.data());

        caffe2::Predictor::TensorVector input_vec{&input};
        _predictor->run(input_vec, &output_vec);

        if (output_vec.capacity() > 0) {
            for (auto output : output_vec) {
                // currently only one dimensional output supported
                result = [NSMutableArray arrayWithCapacity:output_vec.size()];
                NSLog(@"result = %@", result);
                for (auto i = 0; i < output->size(); ++i) {
                    result[i] = @(output->template data<float>()[i]);
                }
            }
        }
        
        
        self.busyWithInference = false;
    }
    
    // When finished, release the context/ data
    CGContextRelease(cgctx);
    if (data) {
        free(data);
    }
    
    self.busyWithInference = false;
    return result;
}

@end
