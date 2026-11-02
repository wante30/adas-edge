#include "../hmi/frame_crc.h"
#include <vector>
#include <cassert>
int main() {
    const unsigned char known[] = "123456789";
    assert(adasFrameCrc(known,9)==0xcbf43926U);
    std::vector<unsigned char> bytes(1843249);
    for (size_t i=0;i<bytes.size();++i) bytes[i]=static_cast<unsigned char>(i*31+i/7);
    for (size_t offset=0;offset<8;++offset) {
        for (size_t length : {size_t(0),size_t(1),size_t(7),size_t(8),size_t(31),size_t(1843200)})
            assert(adasFrameCrc(bytes.data()+offset,length)==::crc32(0,bytes.data()+offset,length));
    }
}
