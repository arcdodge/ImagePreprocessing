Chart.defaults.global.defaultFontFamily = 'Nunito', '-apple-system,system-ui,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif';
Chart.defaults.global.defaultFontColor = '#858796';

function DrawLineChart(summaryListlist) {
    //建立圖表資料
    var datachart = {
        labels: [],
        datasets: [{
            label: '確診人數',
            data: [],
            borderColor: '#FF5376',
            fill: false
        }]
    };

    //將日期與人數push到labels和data陣列中
    for (var i = 0; i < summaryListlist.length; i++) {
        datachart.labels.push(summaryListlist[i].shortDate);
        datachart.datasets[0].data.push(summaryListlist[i].confirmed);
    }

    //繪製圖表
    var ctx = document.getElementById("myChart").getContext("2d");
    var chart = new Chart(ctx, {
        type: 'line', //圖表類型
        data: datachart,
    })
}



var MultiLineChartCtx = new Map();
var MultiLineChartChart = new Map();
var MultiLineChartColors = new Map();
function DrawMultiLineChart(ChartName, lineChartDatas) {
    MultiLineChartColors.clear();
    //建立圖表資料
    var multiLinedatachart = {
        labels: [],
        datasets: []
    };

    for (var i = 0; i < lineChartDatas.length; i++) {
        multiLinedatachart.datasets.push({
            label: lineChartDatas[i].lineName,
            data: [],
            lineTension: 0.3,
            backgroundColor: "rgba(78, 115, 223, 0.05)",
            borderColor: GetColors(lineChartDatas[i].lineName),
            pointRadius: 3,
            pointBackgroundColor: GetColors(lineChartDatas[i].lineName),
            pointBorderColor: GetColors(lineChartDatas[i].lineName),
            pointHoverRadius: 3,
            pointHoverBackgroundColor: GetColors(lineChartDatas[i].lineName),
            pointHoverBorderColor: GetColors(lineChartDatas[i].lineName),
            pointHitRadius: 10,
            pointBorderWidth: 2,
            fill: false
        });
        for (var j = 0; j < lineChartDatas[i].values.length; j++) {
            if (i == 0) {
                multiLinedatachart.labels.push(lineChartDatas[0].xAxisName + ' ' + (j + 1));
            }
            multiLinedatachart.datasets[i].data.push(lineChartDatas[i].values[j]);
        }
    }

    //繪製圖表
    MultiLineChartCtx.set(ChartName, document.getElementById(ChartName).getContext("2d"));

    if ([...MultiLineChartChart.keys()].some(val => val === ChartName)) {
        chart = MultiLineChartChart.get(ChartName);
        chart.destroy();
    }
    MultiLineChartChart.set(ChartName, new Chart(MultiLineChartCtx.get(ChartName), {
        type: 'line', //圖表類型
        data: multiLinedatachart,
        options: {
            maintainAspectRatio: false
        }
    }));

    MultiLineChartChart.get(ChartName).update();
}
function addData(ChartName, id, xAxisName, valueCount, value) {
    if (MultiLineChartChart.get(ChartName).data.labels.length < valueCount) {
        for (var i = MultiLineChartChart.get(ChartName).data.labels.length + 1; i <= valueCount; i++) {
            MultiLineChartChart.get(ChartName).data.labels.push(xAxisName + ' ' + i);
        }
    }
    MultiLineChartChart.get(ChartName).data.datasets[id].data.push(value);
    MultiLineChartChart.get(ChartName).update();
}
function getUniqueRGBA(colors) {
    var randomColor;
    do {
        var r = Math.floor(Math.random() * 256);
        var g = Math.floor(Math.random() * 256);
        var b = Math.floor(Math.random() * 256);
        randomColor = `rgba(${r}, ${g}, ${b}, 1)`;
    } while (colors.indexOf(randomColor) !== -1);
    colors.push(randomColor);
    return randomColor;
}

function GetColors(LineName) {
    if (![...MultiLineChartColors.keys()].some(val => val === LineName))
        MultiLineChartColors.set(LineName, getUniqueRGBA([...MultiLineChartColors.values()]));
    return MultiLineChartColors.get(LineName);
}
