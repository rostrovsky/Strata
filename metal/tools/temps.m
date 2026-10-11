// metal/tools/temps.m - the Mac's temperature sensors, without sudo (metal/tools/bench.py waits with it until the
// Mac is as cool as it was before a run: a fanless Mac is slower once it is warm).
//
//     clang -O2 -fobjc-arc -framework Foundation -framework IOKit metal/tools/temps.m -o temps && ./temps
//
// Prints "name celsius" lines, hottest first.  The sensors are read through IOKit's HID event system, the way
// Activity Monitor-style tools do; nothing is changed on the Mac.
#import <Foundation/Foundation.h>
#include <IOKit/hidsystem/IOHIDEventSystemClient.h>

typedef struct __IOHIDEvent* IOHIDEventRef;
typedef struct __IOHIDServiceClient* IOHIDServiceClientRef;
IOHIDEventSystemClientRef IOHIDEventSystemClientCreate(CFAllocatorRef allocator);
int IOHIDEventSystemClientSetMatching(IOHIDEventSystemClientRef client, CFDictionaryRef match);
IOHIDEventRef IOHIDServiceClientCopyEvent(IOHIDServiceClientRef, int64_t, int32_t, int64_t);
CFStringRef IOHIDServiceClientCopyProperty(IOHIDServiceClientRef service, CFStringRef property);
double IOHIDEventGetFloatValue(IOHIDEventRef event, int32_t field);

int main(void) {
    @autoreleasepool {
        NSDictionary* match = @{@"PrimaryUsagePage": @(0xff00), @"PrimaryUsage": @(5)};
        IOHIDEventSystemClientRef client = IOHIDEventSystemClientCreate(kCFAllocatorDefault);
        IOHIDEventSystemClientSetMatching(client, (__bridge CFDictionaryRef) match);
        NSArray* services = CFBridgingRelease(IOHIDEventSystemClientCopyServices(client));
        NSMutableArray* rows = [NSMutableArray array];
        for (id s in services) {
            IOHIDServiceClientRef sc = (__bridge IOHIDServiceClientRef) s;
            NSString* name = CFBridgingRelease(IOHIDServiceClientCopyProperty(sc, CFSTR("Product")));
            IOHIDEventRef ev = IOHIDServiceClientCopyEvent(sc, 15, 0, 0);   // 15: temperature
            if (ev == NULL) continue;
            const double c = IOHIDEventGetFloatValue(ev, 15 << 16);
            CFRelease(ev);
            if (name != nil && c > 0 && c < 150) [rows addObject:@[name, @(c)]];
        }
        [rows sortUsingComparator:^NSComparisonResult(NSArray* a, NSArray* b) { return [b[1] compare:a[1]]; }];
        for (NSArray* r in rows) printf("%s %.1f\n", [r[0] UTF8String], [r[1] doubleValue]);
    }
    return 0;
}
