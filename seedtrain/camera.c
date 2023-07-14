#pragma warning(disable:4996) //解决scanf函数报错问题
#include "tchar.h"
#include "stdio.h"

// 属性-调试-环境 PATH=C:\Program Files (x86)\Specim\SDKs\SpecSensor\2019_443\bin\x64
// 属性-C/C++ -常规-附加包含目录 C:\Program Files (x86)\Specim\SDKs\SpecSensor\2019_443\include
// 链接器-常规-附加库目录C:\Program Files (x86)\Specim\SDKs\SpecSensor\2019_443\bin\x64
// 链接器-输入-附加依赖项 SpecSensor.lib


//#include "stdafx.h"
#include "SI_sensor.h"
#include "SI_errors.h"
#include "tchar.h"
#include "stdio.h"
#pragma warning(disable:4996) //解决scanf函数报错问题
#define LICENSE_PATH L"C:/Program Files (x86)/Specim/SDKs/SpecSensor/SpecSensor SDK.lic"
int _tmain(int argc, _TCHAR* argv[])
{
	// Create the necessary variables
	int nError = siNoError;
	SI_64 nDeviceCount = 0;
	SI_WC szDeviceName[4096];
	SI_WC szDeviceDescription[4096];
	// Load SpecSensor and get the device count
	SI_CHK(SI_Load(LICENSE_PATH));
	SI_CHK(SI_GetInt(SI_SYSTEM, L"DeviceCount", &nDeviceCount));

	wprintf(L"Device count: %d\n", nDeviceCount);
	// Iterate through each devices to print their name and description
	for (int n = 0; n < nDeviceCount; n++)
	{
		SI_CHK(SI_GetEnumStringByIndex(SI_SYSTEM, L"DeviceName", n, szDeviceName,
			4096));
		SI_CHK(SI_GetEnumStringByIndex(SI_SYSTEM, L"DeviceDescription", n,
			szDeviceDescription, 4096));
		wprintf(L"Device %d:\n", n);
		wprintf(L"\tName: %s\n", szDeviceName);
		wprintf(L"\tDescription: %s\n", szDeviceDescription);
	}
	// Unload the library
	SI_CHK(SI_Unload());
	wprintf(L"\r\nPress any key to quit...\r\n");
	getchar();

Error:
	return 0;
}







//#include "stdafx.h"
#include <iostream>
#include <windows.h>
#include "SI_sensor.h"
#include "SI_errors.h"
#include "tchar.h"
#pragma warning(disable:4996) //解决scanf函数报错问题
#define LICENSE_PATH L"C:/Program Files (x86)/Specim/SDKs/SpecSensor/SpecSensor SDK.lic"
SI_H g_hDevice = 0;
//define SI_IMPEXP_CONV __cdecl  C语言默认的函数调用方法：所有参数从右到左依次入栈，这些参数由调用者清除，称为手动清栈
int SI_IMPEXP_CONV onDataCallback(SI_U8* _pBuffer, SI_64 _nFrameSize, SI_64 _nFrameNumber, void* _pContext);  //这行其实可以注释掉



int SelectDevice(void)
{
	int nError = siNoError;  // siNoError=0
	SI_64 nDeviceCount = 0;  //typedef long long SI_64
	SI_WC szDeviceName[4096];//typedef wchar_t SI_WC    char占一个字节，wchar_t（宽字符）占两个字节
	int nIndex = -1;
	SI_CHK(SI_GetInt(SI_SYSTEM, L"DeviceCount", &nDeviceCount)); //define SI_SYSTEM=0
	wprintf(L"Device count: %d\n", nDeviceCount); //wprintf同printf
	// Iterate through each devices to print their name
	for (int n = 0; n < nDeviceCount; n++)
	{
		SI_CHK(SI_GetEnumStringByIndex(SI_SYSTEM, L"DeviceName", n, szDeviceName,4096));//SI_GetEnumStringByIndex:根据指定特性的字符串设置当前使用的索引
		wprintf(L"\t%d: %s\n", n, szDeviceName);
	}
	// Select a device
	wprintf(L"Select a device: ");
	scanf_s("%d", &nIndex);//scanf_s:很多带“_s”后缀的函数是为了让原版函数更安全
	if ((nIndex >= nDeviceCount) || (nIndex == -1))
	{
		wprintf(L"Invalid index");
		return -1;
	}
Error:
	return nIndex;
}

int SI_IMPEXP_CONV FeatureCallback1(SI_H Hndl, SI_WC* Feature, void* Context)
{
	if (wcscmp(Feature, L"Camera.ExposureTime") == 0)  //wcscmp():比较字符串是否相等
	{
		wprintf(L"FeatureCallback1: Camera.ExposureTime\n");
	}
	else if (wcscmp(Feature, L"Camera.FrameRate") == 0)
	{
		wprintf(L"FeatureCallback1: Camera.FrameRate\n");
	}
	return 0;
}
int SI_IMPEXP_CONV FeatureCallback2(SI_H Hndl, SI_WC* Feature, void* Context)
{
	if (wcscmp(Feature, L"Camera.ExposureTime") == 0)
	{
		wprintf(L"FeatureCallback2: Camera.ExposureTime\n");
	}
	return 0;
}
int SI_IMPEXP_CONV onDataCallback(SI_U8* _pBuffer, SI_64 _nFrameSize, SI_64 _nFrameNumber, void* _pContext)
{
	wprintf(L"%d ", _nFrameNumber);
	return 0;
}

//主函数
int _tmain(int argc, _TCHAR* argv[])
{
	// Create the necessary variables
	int nError = siNoError;
	int nDeviceIndex = -1;
	int nAction = 0;
	wchar_t szMessage[] = L"Select an action:\n\t0: exit\n\t1: startacquisition\n\t2: stop acquisition\n";
		// Load SpecSensor and get the device count
	wprintf(L"Loading SpecSensor...\n");
	SI_CHK(SI_Load(LICENSE_PATH));// the first code
	// Select a device
	nDeviceIndex = SelectDevice();

	// Open the device and set the callbacks
	SI_CHK(SI_Open(nDeviceIndex, &g_hDevice));
	SI_CHK(SI_Command(g_hDevice, L"Initialize"));
	SI_CHK(SI_RegisterFeatureCallback(g_hDevice, L"Camera.FrameRate",FeatureCallback1, 0));
	SI_CHK(SI_RegisterFeatureCallback(g_hDevice, L"Camera.ExposureTime",FeatureCallback1, 0));
	SI_CHK(SI_RegisterFeatureCallback(g_hDevice, L"Camera.ExposureTime",FeatureCallback2, 0));
	SI_CHK(SI_RegisterDataCallback(g_hDevice, onDataCallback, 0));
	// Prompt commands
	wprintf(L"%s", szMessage);
	while (scanf("%d", &nAction))
	{
		if (nAction == 0)
		{
			wprintf(L"Bye bye!");
			break;
		}
		else if (nAction == 1)
		{
			wprintf(L"Start acquisition");
			SI_CHK(SI_Command(g_hDevice, L"Acquisition.Start"));
		}
		else if (nAction == 2)
		{
			wprintf(L"Stop acquisition");
			SI_CHK(SI_Command(g_hDevice, L"Acquisition.Stop"));
		}
		wprintf(L"%s", szMessage);
	}

Error:
	if (SI_FAILED(nError))
	{
		wprintf(L"An error occurred: %s\n", SI_GetErrorString(nError));
	}
	SI_Close(g_hDevice);
	SI_Unload();
	return SI_FAILED(nError) ? -1 : 0;
}



