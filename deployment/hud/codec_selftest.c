#include "hud_v1.h"
#include "adas_status_v1.h"
#include <assert.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    static const uint32_t seq[9]={1,2,3,4,5,6,7,8,9};
    static const uint32_t state[9]={0x101,0x103,0x105,0x109,0x1c1,0x111,0x121,0,0};
    static const unsigned blink[9]={0,1,2,4,0,0,0,0,0};
    static const uint32_t expected[9]={0x11101,0x21303,0x31505,0x41909,0x511c1,0x61111,0x71121,0x81000,0x90000};
    struct adst_v1 source={0x12345678,3,UINT64_C(0x1122334455667788),UINT64_C(0x0102030405060708),3,2,1,3,26214,1,31,128,750};
    struct adst_v1 decoded;
    uint8_t packet[ADST_V1_SIZE];
    uint8_t reencoded[ADST_V1_SIZE];
    uint32_t word=0;
    unsigned i;
    for (i=0;i<9;++i) {
        assert(!hud_v1_encode(seq[i], i != 8, state[i], blink[i], &word));
        assert(word==expected[i]);
        printf("HUD %08x\n",(unsigned)word);
    }
    assert(hud_v1_encode(65536,1,0x101,0,&word));
    assert(hud_v1_encode(1,1,0x101,1,&word));
    /* Exhaust all state/blink combinations, including disabled blink inputs. */
    for (i=0;i<4096;++i) {
        uint32_t bits=i & 511u;
        unsigned mode=i >> 9;
        int valid=(!(mode & 1) || (bits & 2)) && (!(mode & 2) || (bits & 4))
                  && (!(mode & 4) || (bits & 8));
        int code=hud_v1_encode(42,1,bits,mode,&word);
        assert((code==0)==valid);
        if (!code) assert(word==((42u << 16) | (1u << 12) | (mode << 9) | bits));
    }
    assert(!adst_v1_encode(&source,packet));
    assert(!adst_v1_decode(packet,sizeof(packet),&decoded));
    assert(decoded.source_frame_id==source.source_frame_id && decoded.sequence==source.sequence);
    assert(!adst_v1_encode(&decoded,reencoded));
    assert(!memcmp(packet,reencoded,sizeof(packet)));
    printf("ADST ");
    for(i=0;i<ADST_V1_SIZE;++i) printf("%02x",packet[i]);
    printf("\n");
    packet[20]^=1;
    assert(adst_v1_decode(packet,sizeof(packet),&decoded));
    assert(!adst_v1_encode(&source,packet));
    packet[52]=1; /* Reserved byte is invalid even with a correct CRC. */
    adst_put(packet+56,adst_crc32(packet,56),4);
    assert(adst_v1_decode(packet,sizeof(packet),&decoded));
    assert(!adst_v1_encode(&source,packet));
    adst_put(packet+36,65535,2);
    adst_put(packet+56,adst_crc32(packet,56),4);
    assert(adst_v1_decode(packet,sizeof(packet),&decoded));
    if (argc==2) {
        FILE *f=fopen(argv[1],"rb");
        assert(f);
        assert(fread(packet,1,sizeof(packet),f)==sizeof(packet));
        assert(fgetc(f)==EOF);
        assert(!fclose(f));
        assert(!adst_v1_decode(packet,sizeof(packet),&decoded));
        assert(!adst_v1_encode(&decoded,reencoded));
        assert(!memcmp(packet,reencoded,sizeof(packet)));
        printf("PYTHON_DECODE %u %u %llu\n",(unsigned)decoded.sequence,(unsigned)decoded.stream_id,(unsigned long long)decoded.source_frame_id);
    }
    puts("NATIVE_C_SELFTEST_PASS");
    return 0;
}
