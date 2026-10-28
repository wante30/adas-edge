from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_item.cpp')
s=p.read_text(encoding='utf-8')
# add verify flag after path
s=s.replace('''    const std::string path = pgvFile.toStdString();
    std::array<unsigned char, kPgvHeaderBytes> header{};''','''    const std::string path = pgvFile.toStdString();
    // CRC is mandatory by default. The local PGVF publisher uses atomic
    // rename, so a faster local-only mode can skip CRC after header/length
    // validation; real PCIe keeps CRC unless explicitly opted out.
    const bool verifyCrc = qEnvironmentVariableIntValue("QT_PGVF_SKIP_CRC") == 0;
    std::array<unsigned char, kPgvHeaderBytes> header{};''',1)
# replace ifstream open/read block
old='''        std::ifstream input(path, std::ios::binary);
        if (!input) {
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
            continue;
        }
        input.read(reinterpret_cast<char *>(header.data()), static_cast<std::streamsize>(header.size()));
        if (input.gcount() != static_cast<std::streamsize>(header.size())) {
            source_crc_errors_.fetch_add(1);
            previous = info;
            havePrevious = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }'''
new='''        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
            continue;
        }
        const ssize_t headerGot = ::pread(fd, header.data(), header.size(), 0);
        if (headerGot != static_cast<ssize_t>(header.size())) {
            ::close(fd);
            source_crc_errors_.fetch_add(1);
            previous = info;
            havePrevious = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }'''
if old not in s: raise SystemExit('header read marker')
s=s.replace(old,new,1)
# replace payload read + crc block
old='''        input.read(reinterpret_cast<char *>(slot->pixels.data()), static_cast<std::streamsize>(kFrameBytes));
        previous = info;
        havePrevious = true;
        if (input.gcount() != static_cast<std::streamsize>(kFrameBytes) ||
            crc32Bytes(slot->pixels.data(), kFrameBytes) != expectedCrc) {
            source_crc_errors_.fetch_add(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }'''
new='''        const ssize_t payloadGot = ::pread(fd, slot->pixels.data(), kFrameBytes, kPgvHeaderBytes);
        ::close(fd);
        previous = info;
        havePrevious = true;
        if (payloadGot != static_cast<ssize_t>(kFrameBytes) ||
            (verifyCrc && crc32Bytes(slot->pixels.data(), kFrameBytes) != expectedCrc)) {
            source_crc_errors_.fetch_add(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }'''
if old not in s: raise SystemExit('payload read marker')
s=s.replace(old,new,1)
p.write_text(s,encoding='utf-8')
print('producer pread/optional crc')
