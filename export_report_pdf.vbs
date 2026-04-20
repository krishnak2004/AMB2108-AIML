Dim wordApp
Dim doc
Dim inputPath
Dim outputPath

inputPath = "C:\Users\Shreya\OneDrive\ドキュメント\New project\Weed_Detection_Random_Forest_Report.html"
outputPath = "C:\Users\Shreya\OneDrive\ドキュメント\New project\Weed_Detection_Random_Forest_Report.pdf"

On Error Resume Next

Set wordApp = CreateObject("Word.Application")
If Err.Number <> 0 Then
    WScript.Echo "ERROR: Unable to start Word - " & Err.Description
    WScript.Quit 1
End If

wordApp.Visible = False
wordApp.DisplayAlerts = 0

Set doc = wordApp.Documents.Open(inputPath, False, True)
If Err.Number <> 0 Then
    WScript.Echo "ERROR: Unable to open HTML report - " & Err.Description
    wordApp.Quit
    WScript.Quit 1
End If

doc.ExportAsFixedFormat outputPath, 17
If Err.Number <> 0 Then
    WScript.Echo "ERROR: Unable to export PDF - " & Err.Description
    doc.Close False
    wordApp.Quit
    WScript.Quit 1
End If

doc.Close False
wordApp.Quit

WScript.Echo "PDF exported to: " & outputPath
