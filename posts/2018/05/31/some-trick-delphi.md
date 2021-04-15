---
title: Delphi) Some tricks
date: 2018-05-31 08:11:51
published: true
tags:
  - programming
  - delphi
description: '첫 직장에서 델파이로 개발했을 때 유용하게 사용하던 코드 모음.'
category: programming
slug: /2018/05/31/some-trick-delphi/
template: post
---

정보: github page code highlight는 델파이를 지원하지 않는다.

## checksum

```pascal
function GetCheckSum(FileName: string): DWORD;
var
  F: file of DWORD;
  P: Pointer;
  Fsize: DWORD;
  Buffer: array [0..500] of DWORD;
begin
  FileMode := 0;
  AssignFile(F, FileName);
  Reset(F);
  Seek(F, FileSize(F) div 2);
  Fsize := FileSize(F) - 1 - FilePos(F);
  if Fsize > 500 then Fsize := 500;
  BlockRead(F, Buffer, Fsize);
  Close(F);
  P := @Buffer;
  asm
     xor eax, eax
     xor ecx, ecx
     mov edi , p
     @again:
       add eax, [edi + 4*ecx]
       inc ecx
       cmp ecx, fsize
     jl @again
     mov @result, eax
  end;
end;
```

## 프로세스 실행 중인지 확인

```pascal
function CheckProcessRunning(exeFileName: string): Boolean;
var
  ContinueLoop: BOOL;
  FSnapshotHandle: THandle;
  FProcessEntry32: TProcessEntry32;
begin
  FSnapshotHandle := CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
  FProcessEntry32.dwSize := SizeOf(FProcessEntry32);
  ContinueLoop := Process32First(FSnapshotHandle, FProcessEntry32);
  Result := False;
  while Integer(ContinueLoop) <> 0 do
  begin
    if ((UpperCase(ExtractFileName(FProcessEntry32.szExeFile)) =
      UpperCase(ExeFileName)) or (UpperCase(FProcessEntry32.szExeFile) =
      UpperCase(ExeFileName))) then
    begin
      Result := True;
    end;
    ContinueLoop := Process32Next(FSnapshotHandle, FProcessEntry32);
  end;
  CloseHandle(FSnapshotHandle);
end;
```

## 프로세스 죽이기

```pascal
function StopProcess(ExeFileName: string) : Integer;
const
  PROCESS_TERMINATE = $0001;
var
  ContinueLoop: BOOL;
  FSnapshotHandle: THandle;
  FProcessEntry32: TProcessEntry32;
begin
  Result := 0;
  FSnapshotHandle := CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
  FProcessEntry32.dwSize := SizeOf(FProcessEntry32);
  ContinueLoop := Process32First(FSnapshotHandle, FProcessEntry32);
  while Integer(ContinueLoop) <> 0 do
  begin
    if ((UpperCase(ExtractFileName(FProcessEntry32.szExeFile)) =
      UpperCase(ExeFileName)) or (UpperCase(FProcessEntry32.szExeFile) =
      UpperCase(ExeFileName)))
    then
      Result := Integer(TerminateProcess(
                        OpenProcess(PROCESS_TERMINATE,
                                    BOOL(0),
                                    FProcessEntry32.th32ProcessID),
                                    0));
     ContinueLoop := Process32Next(FSnapshotHandle, FProcessEntry32);
  end;
  CloseHandle(FSnapshotHandle);
end;
```

## string to html (html parsing)

```pascal
procedure TForm2.ParsingHTML;
var
  IdHTTP: TIdHTTP;
  Stream: TStringStream;
begin
  IdHTTP := TIdHTTP.Create(nil);
  Stream := TStringStream.Create;
  try
    try
      IdHTTP.Get(edt1.Text, Stream);
      JenkinsParsingResult := Stream.DataString;
    except
      //exception Raise... (올바르지 않은 주소일때)
    end;
  finally
    tmr1.Enabled := True;
    Stream.Free;
    IdHTTP.Free;
  end;
end;
```

## File Drag & drop (파일 드래그 앤 드랍)

in formcreate

```pascal
DragAcceptFiles(Handle, True);
```

```pascal
procedure TForm2.WMDROPFILES(var Msg: TWMDropFiles);
var
  i, amount: Integer;
  FileName: array[0..MAX_PATH] of Char;
begin
  inherited;
  try
    Amount := DragQueryFile(Msg.Drop, $FFFFFFFF, FileName, MAX_PATH);
    for i := 0 to (Amount - 1) do
    begin
      DragQueryFile(Msg.Drop, i, FileName, MAX_PATH);
     //FileName에 파일/폴더 path가 들어옵니다.
  finally
    DragFinish(Msg.Drop);
  end;
end;
```

## get md5

```pascal
function GetCheckSum(FileName: string): string;
var
  IdMD5: TIdHashMessageDigest5;
  FS: TFileStream;
begin
   IdMD5 := TIdHashMessageDigest5.Create;
   FS := TFileStream.Create(FileName, fmOpenRead or fmShareDenyWrite);
   try
     Result := IdMD5.HashStreamAsHex(FS);
   finally
     FS.Free;
     IdMD5.Free;
   end;
end;
```

## 작업표시줄 제어

### auto hide를 always show로 변경

```pascal
var
  ABData: TAppBarData;
begin
  ABData.cbSize := SizeOf(TAppBarData);
  ABData.hWnd := FindWindow('SHELL_TRAYWND', nil);
  ABData.lParam := LParam(0);
  SHAppBarMessage($0000000a, ABData);
```

### 작업표시줄 숨기기. 숨기는게 아니고 레알루다가 없앰

```pascal
var
  hTaskbar: THandle;
begin
  hTaskbar := FindWindow('Shell_TrayWnd', Nil);
  ShowWindow(hTaskbar, SW_HIDE);
```

### 작업표시줄 보이기

```pascal
var
  hTaskbar: THandle;
begin
  hTaskbar := FindWindow('Shell_TrayWnd', Nil);
  ShowWindow(hTaskbar, SW_SHOWNORMAL);
```

## 인터넷 연결상태 확인

```pascal
function CheckInternetConnection(): Boolean;
const
  // Local system has a valid connection to the Internet, but it might or might
  // not be currently connected.
  INTERNET_CONNECTION_CONFIGURED = $40;

  // Local system uses a local area network to connect to the Internet.
  INTERNET_CONNECTION_LAN = $02;

  // Local system uses a modem to connect to the Internet
  INTERNET_CONNECTION_MODEM = $01;

  // Local system is in offline mode.
  INTERNET_CONNECTION_OFFLINE = $20;

  // Local system uses a proxy server to connect to the Internet
  INTERNET_CONNECTION_PROXY = $04;

  // Local system has RAS installed.
  INTERNET_RAS_INSTALLED = $10;

var
  InetState: DWORD;
  hHttpSession, hReqUrl: HInternet;
begin
  Result:= InternetGetConnectedState(@InetState, 0);
  if (
    Result
    and
    (
      InetState and INTERNET_CONNECTION_CONFIGURED
        = INTERNET_CONNECTION_CONFIGURED)
    ) then
  begin
    // so far we ONLY know there's a valid connection. See if we can grab some
    // known URL ...
    hHttpSession:= InternetOpen(
      PChar(Application.Title), // this line is the agent string
      INTERNET_OPEN_TYPE_PRECONFIG, nil, nil, 0
    );
    try
      hReqUrl:= InternetOpenURL(
        hHttpSession,
        PChar('http://wwww.example.com'{ the URL to check }),
        nil,
        0,
        0,
        0
      );
      Result := hReqUrl <> nil;
      InternetCloseHandle(hReqUrl);
    finally
      InternetCloseHandle(hHttpSession);
    end;
  end
  else
    if (
      InetState and INTERNET_CONNECTION_OFFLINE = INTERNET_CONNECTION_OFFLINE
    ) then
      Result := False; // we know for sure we are offline.
end;
```
