;----------------------------------------------
; This is the Mailpile installer for Windows

!define PRODUCT_NAME "Mailpile"
!define PRODUCT_VERSION "Beta"
!define PRODUCT_PUBLISHER "Mailpile ehf"
!define PRODUCT_WEB_SITE "https://www.mailpile.is/"
!define PRODUCT_EXE_NAME "mailpile.exe"

!include "MUI.nsh"
!define MUI_ABORTWARNING
!define MUI_ICON "packages\windows\mailpile.ico"
!define MUI_UNICON "${NSISDIR}\Contrib\Graphics\Icons\modern-uninstall.ico"
!define MUI_FINISHPAGE_SHOWREADME ""
!define MUI_FINISHPAGE_SHOWREADME_NOTCHECKED
!define MUI_FINISHPAGE_SHOWREADME_TEXT "Create Desktop Shortcut"
!define MUI_FINISHPAGE_SHOWREADME_FUNCTION createDesktopShortcut

; Wizard pages
!insertmacro MUI_PAGE_LICENSE "COPYING.md"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"


Name "${PRODUCT_NAME} ${PRODUCT_VERSION}"
InstallDir "$PROGRAMFILES\${PRODUCT_PUBLISHER}\${PRODUCT_NAME}"
ShowInstDetails show
ShowUnInstDetails show

OutFile "Mailpile-Installer.exe"

;Get installation folder from registry if available
InstallDirRegKey HKCU "Software\Mailpile" ""

;Request application privileges for Windows Vista
RequestExecutionLevel admin


Section "install" InstallationInfo
	SetOutPath $INSTDIR
	SetOverwrite ifnewer

	File /r /x junk /x macosx /x tmp /x .git /x testing "*.*"

	WriteRegStr HKCU "Software\Mailpile" "" "$INSTDIR"

	createDirectory "$SMPROGRAMS\Mailpile"
	createShortCut "$SMPROGRAMS\Mailpile\Start Mailpile.lnk" "$INSTDIR\Mailpile.exe" "" "$INSTDIR\packages\windows\mailpile.ico"
	createShortCut "$SMPROGRAMS\Mailpile\Uninstall Mailpile.lnk" "$INSTDIR\uninstall.exe" "" ""
	WriteINIStr "$SMPROGRAMS\Mailpile\Open Mailpile.url" "InternetShortcut" "URL" "http://localhost:33411"

; This would start Mailpile automatically, not sure we're ready for that
;	WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Run" "${PRODUCT_NAME}" "$INSTDIR\${PRODUCT_EXE_NAME}"

	WriteUninstaller $INSTDIR\uninstall.exe
	WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}" "DisplayName" "${PRODUCT_NAME} (remove only)"
	WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}" "UninstallString" "$INSTDIR\uninstall.exe"
	WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}" "DisplayVersion" "${PRODUCT_VERSION}"
	WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}" "URLInfoAbout" "${PRODUCT_WEB_SITE}"
	WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}" "Publisher" "${PRODUCT_PUBLISHER}"
SectionEnd

Section "un.Uninstall"
; We list things which the /r would catch anyway, to make the progress bar
; a bit more interesting to the poor impatient user who can't wait to get
; rid of us... ;-)
	Delete "$DESKTOP\${PRODUCT_NAME}.lnk"
	DeleteRegKey HKCU "Software\Mailpile"
	DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}"
	RMDir /r "$SMPROGRAMS\Mailpile"
	RMDir /r "$INSTDIR\Python27\DLLs"
	RMDir /r "$INSTDIR\Python27\Lib\site-packages"
	RMDir /r "$INSTDIR\Python27\Lib"
	RMDir /r "$INSTDIR\Python27\libs"
	RMDir /r "$INSTDIR\Python27\include"
	RMDir /r "$INSTDIR\Python27"
	RMDir /r "$INSTDIR\OpenSSL"
	RMDir /r "$INSTDIR\GnuPG"
	RMDir /r "$INSTDIR\locale"
	RMDir /r "$INSTDIR\mailpile"
	RMDir /r "$INSTDIR\plugins"
	RMDir /r "$INSTDIR"
	SetAutoClose true
SectionEnd

Function un.onUninstSuccess
	HideWindow
	MessageBox MB_ICONINFORMATION|MB_OK "Mailpile has been successfully removed from your computer. How sad..."
FunctionEnd

Function createDesktopShortcut
	CreateShortCut "$DESKTOP\${PRODUCT_NAME}.lnk" "$INSTDIR\${PRODUCT_EXE_NAME}" ""
FunctionEnd
