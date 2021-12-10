#include <sys/stat.h>
#include <dirent.h>
#include <string>
#include <iostream>
 
void traverseFolder(const char* pInputPath) {
    DIR* dir = opendir(pInputPath);
    if (!dir) {
        printf("No such dir!\n");
        return;
    }
    char name[512];
    
    struct dirent *pEntry;
    while ((pEntry = readdir(dir))) {
        // 跳过. .. 等 和文件夹
        if((strncmp(pEntry->d_name, ".", 1) == 0) || (pEntry->d_type & DT_DIR)){
            continue;
        }
        
        memset(name, 0, sizeof(name));
        if(pEntry->d_type & DT_REG) {
            std::cout << pEntry->d_name << std::endl;
        }
    }
}
int main() {
    traverseFolder("../imgs/test1_combined_imgs");
    return 0;
}