# w = "574528ee 5d2aa02a 4cf15010 52a214fa 32e7ba61 2d357b57 184882df 47a35144 4d75bd23 2a0eaeec 2f31ba80 217dedaf 1a861652 0129747a 3d4f71a6 1e565b6a 12715a44 5c692df2 1f6f4752 65504cc9 4e5785aa 042408ac 2ab32c6b 25521f4a 16a4fb2f 4f3509ca 0ca1a2ad 584a5056 2a39fb9d 52a99e65 2a827cbc 017f248c 2fd43e8f 7773cccd 54213986 62bbf8f0"
# x = "7fdcc233 1befd79f 41a7c4c9 6b68079a 4e6afb66 25e45d32 519b500d 431bd7b7 3f2dba31 7c83e458 257130a3 62bbd95a 189a769b 54e49eb4 71f32454 2ca88611 0836c40e 02901d82 3a95f874 08138641 1e7ff521 7c3dbd3d 737b8ddc 6ceaf087 2463b9ea 5e884adc 51ead36b 2d517796 580bd78f 153ea438 3855585c 70a64e2a 6a2342ec 2a487cb0 1d4ed43b 725a06fb"

w = "574528ee 5d2aa02a 4cf15010 52a214fa 32e7ba61 2d357b57 184882df 47a35144 4d75bd23 2a0eaeec 2f31ba80 217dedaf 1a861652 0129747a 3d4f71a6 1e565b6a 12715a44 5c692df2 1f6f4752 65504cc9 4e5785aa 042408ac 2ab32c6b 25521f4a 16a4fb2f 4f3509ca 0ca1a2ad 584a5056 2a39fb9d 52a99e65 2a827cbc 017f248c 2fd43e8f 6c69bc65 139cf2f6 6d2bd373"
x = "7fdcc233 1befd79f 41a7c4c9 6b68079a 4e6afb66 25e45d32 519b500d 431bd7b7 3f2dba31 7c83e458 257130a3 62bbd95a 189a769b 54e49eb4 71f32454 2ca88611 0836c40e 02901d82 3a95f874 08138641 1e7ff521 7c3dbd3d 737b8ddc 6ceaf087 2463b9ea 5e884adc 51ead36b 2d517796 580bd78f 153ea438 3855585c 70a64e2a 6a2342ec 2a487cb0 1d4ed43b 725a06fb"


map = {'0':'0000', '1':'0001', '2':'0010', '3': '0011', '4':'0100', '5':'0101', '6':"0110", '7':'0111', '8':'1000', '9':'1001', 'a':'1010', 'b':'1011', 'c':'1100', 'd':'1101', 'e':'1110', 'f':'1111'}


val = 0
for i in range(len(x)):
    if w[i] == ' ':
        continue
    tmp0 = map[w[i]]
    tmp1 = map[x[i]]
    for j in range(4):
        val += int(tmp0[j])*int(tmp1[j])

print("y00: ", val)

# val = 0
# for i in range(len(x0)):
#     val += int(x0[i])*int(y1[i])
# print("y01: ", val)

# val = 0
# for i in range(len(x0)):
#     val += int(x1[i])*int(y0[i])
# print("y10: ", val)

# val = 0
# for i in range(len(x0)):
#     val += int(x1[i])*int(y1[i])
# print("y11: ", val)

print(len(w))