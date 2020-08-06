---

layout: single
title:  "Linux, KakaoTalk, yeees..."
header:
  teaser: ""
categories: 
  - Linux
tags:
  - KakaoTalk, Ubuntu, 18.04
comments: True

---

# 리눅스(우분투), 카카오톡, 성공적

리눅스를 쓰면서 가장 불편한 점을 뽑아보자면 **카카오톡**을 못쓰는 점이 아닐까 싶다..

무슨 말이 필요있을까

그냥 순서만 따라해주세요.

가시죠


1. 나눔 폰트 설치

    이후에 카카오톡에서 폰트 변경 가능

    ```bash
    $ sudo apt-get install fonts-nanum*
    ```

2. wine 설치 (5.0버전 이상)

    윈도우에서 사용하는 프로그램들을 리눅스에서 사용할 수 있게 해주는 툴

    카카오톡 같은 경우 최신버전은 지원하지 않지만 XP에서 사용하던 버전인 구형버전을 설치

    이곳을 참고

    [How to Install Wine 5.0 Stable in Ubuntu 18.04, 19.10](http://ubuntuhandbook.org/index.php/2020/01/install-wine-5-0-stable-ubuntu-18-04-19-10/)

    1. Wine Dependency 설치

        ```bash
        $ sudo apt-get install libgnutls30:i386 libldap-2.4-2:i386 libgpg-error0:i386 libxml2:i386 libasound2-plugins:i386 libsdl2-2.0-0:i386 libfreetype6:i386 libdbus-1-3:i386 libsqlite3-0:i386
        ```

    2. 32비트 아키텍쳐 사용가능

        ```bash
        $ sudo dpkg --add-architecture i386
        ```

    3. 레포지토리키 다운로드 및 설치

        ```bash
        $ wget -nc https://dl.winehq.org/wine-builds/winehq.key; sudo apt-key add winehq.key
        ```

    4. wine 레포지토리 추가 (18.04버전)

        +. **bionic → 18.04**, xenial → 16.04

        ```bash
        $ sudo apt-add-repository 'deb https://dl.winehq.org/wine-builds/ubuntu/ bionic main'
        ```

    5. Add PPA for the requied `libfaudio0` library (?? 그냥 따라했음)

        ```bash
        $ sudo add-apt-repository ppa:cybermax-dexter/sdl2-backport
        ```

    6. wine 설치

        ```bash
        $ sudo apt update && sudo apt install --install-recommends winehq-stable
        ```

    7. winetricks 설치

        와인

        참고링크

        [우분투 카카오톡 설치 - HiSEON](https://hiseon.me/linux/ubuntu/ubuntu-kakaotalk/)

        ```bash
        $ wget  https://raw.githubusercontent.com/Winetricks/winetricks/master/src/winetricks
        $ chmod +x winetricks
        $ ./winetricks --optout
        ```

        창이 무엇인가 뜬다면 완료, 창을 꺼준다.

3. PlayOnLinux 설치

    참고링크

    [Downloads](https://www.playonlinux.com/en/download.html)

    18.04버전 (Bionic)기준

    ```bash
    $ wget -q "http://deb.playonlinux.com/public.gpg" -O- | sudo apt-key add -
    $ sudo wget http://deb.playonlinux.com/playonlinux_bionic.list -O /etc/apt/sources.list.d/playonlinux.list
    $ sudo apt-get update
    $ sudo apt-get install playonlinux
    ```

4. 카카오톡 설치

    참고링크

    [우분투 16.04, 카카오톡 설치시 xp 버전이라 뜨는 문제 해결방법](https://medium.com/@onlytojay/%EC%9A%B0%EB%B6%84%ED%88%AC-16-04-%EC%B9%B4%EC%B9%B4%EC%98%A4%ED%86%A1-%EC%84%A4%EC%B9%98%EC%8B%9C-xp-%EB%B2%84%EC%A0%84%EC%9D%B4%EB%9D%BC-%EB%9C%A8%EB%8A%94-%EB%AC%B8%EC%A0%9C-%ED%95%B4%EA%B2%B0%EB%B0%A9%EB%B2%95-24c6135fae9d)

    1. 카카오톡 구버전 설치

        [https://kbench.com/software/?q=node/70275](https://kbench.com/software/?q=node/70275)

    2. 위 링크에서 하는 것과 같이 play on linux 에서 설치를 진행해준다.

        +. play on linux를 실행시킬때 "xterm"이 없다고 나오기 때문에 설치해준다

        ```bash
        $ sudo apt install xterm
        ```

        +. 처음에 카카오톡을 설치할 때 글자가 다 깨져서 나옴
        **설치언어가 3가지가 나오는데 그 중에 맨 아래 언어를 선택해서 한국어로 설치**
    
        <font color='red'>**네모가 나오더라도 당황하지 않는다.**</font>

5. 한글화
    1. 굴림체 설치

        윈도우 PC에서 옮겨와도 된다```C:\\Windows\Fonts```경로에 있다.

        근데 구글링이 더 빠르다. 진짜 바로 나온다.

    2. wine 한글 설정
        - 다운받은 굴림체를 다음 폴더에 복사

            ```~/.wine/drive_c/windows/Fonts```

            ```bash
        $ cp Gulim.ttf ~/.wine/drive_c/windows/Fonts
            ```
        
        - 파일 수정

            ```bash
        $ sudo nano ~/.wine/system.reg
            ```

            - Ctrl + w (찾기) : 'MS Shell' + 엔터

                ![1](https://user-images.githubusercontent.com/48716219/89537590-833e0400-d834-11ea-91cc-fa5fdbe355e8.png)

            - 다음과 같이 수정
        
                "MS Shell Dlg", "MS Shell Dlg 2" 를 "Gulim"으로 설정
        
            - Ctrl + O (저장)
            - Ctrl + X (나가기)
        
    3. PlayOnLinux 한글 설정
        - 경로에 폰트 복사

            '~/.PlayOnLinux/wineprefix/[virtual_name]/drive_c/window/Fonts'

            ```bash
        $ cp Gulim.ttf ~/.PlayOnLinux/wineprefix/[~~]/drive_c/windows/Fonts
            ```
        
        - 파일 수정

            ```bash
        $ sudo nano ~/.PlayOnLinux/wineprefix/[virtual_name]/system.reg
            ```
        
            위에서 한것과 동일하게 진행
        
    4. KakaoTalk.desktop 설정

        ```bash
        $ sudo nano ~/Desktop/KakaTalk.desktop
        ```

        ```shell
    Exec=/'무슨무슨 경로경로' %F
        ```

        이 부분을 다음과 같이 바꾼다.

        ```shell
        Exec=/~~~~
    LANG=ko_KR.UTF-8 %F
        ```
        
        ![2](https://user-images.githubusercontent.com/48716219/89537631-918c2000-d834-11ea-8a27-0b291e895772.png))

