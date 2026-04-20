param(
    [string]$Root = (Get-Location).Path,
    [string]$HtmlName = "Final_Submission_Report.html",
    [string]$DocxName = "Weed_Detection_Final_Submission.docx",
    [string]$PdfName = "Weed_Detection_Final_Submission.pdf"
)

$ErrorActionPreference = "Stop"

$htmlPath = (Resolve-Path (Join-Path $Root $HtmlName)).Path
$docxPath = Join-Path $Root $DocxName
$pdfPath = Join-Path $Root $PdfName

$wdAlertsNone = 0
$wdFormatDocumentDefault = 16
$wdExportFormatPDF = 17

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = $wdAlertsNone

try {
    $document = $word.Documents.Open($htmlPath)
    $document.SaveAs2($docxPath, $wdFormatDocumentDefault)
    $document.ExportAsFixedFormat($pdfPath, $wdExportFormatPDF)
    $document.Close()
} finally {
    $word.Quit()
}

Write-Host "Saved report files:"
Write-Host "- $docxPath"
Write-Host "- $pdfPath"
