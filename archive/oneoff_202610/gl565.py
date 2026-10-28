from pathlib import Path
p=Path(r'C:\Users\JingyuWang\adas-edge\deployment\lubancat\qt_gpu_probe\video_item.cpp')
s=p.read_text(encoding='utf-8')
s=s.replace('''                f->glBindTexture(GL_TEXTURE_2D, texture_);
                f->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);''','''                f->glActiveTexture(GL_TEXTURE0);
                f->glBindTexture(GL_TEXTURE_2D, texture_);
                f->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);''',1)
s=s.replace('''f->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, kSourceWidth, kSourceHeight, 0,
                                GL_RGB, GL_UNSIGNED_SHORT_5_6_5, frame->pixels.data());''','''f->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB565, kSourceWidth, kSourceHeight, 0,
                                GL_RGB, GL_UNSIGNED_SHORT_5_6_5, frame->pixels.data());''')
p.write_text(s,encoding='utf-8')
print('565 applied', 'GL_RGB565' in s)
