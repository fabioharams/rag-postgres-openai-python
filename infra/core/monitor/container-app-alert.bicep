metadata description = 'Creates an Azure Monitor metric alert for Container App running status.'

param name string
param location string = 'global'
param tags object = {}

@description('Resource ID of the Container App to monitor')
param containerAppResourceId string

@description('Alert severity (0=Critical, 1=Error, 2=Warning, 3=Informational, 4=Verbose)')
@allowed([ 0, 1, 2, 3, 4 ])
param severity int = 1

@description('Whether the alert rule is enabled')
param enabled bool = true

resource runningStatusAlert 'Microsoft.Insights/metricAlerts@2018-03-01' = {
  name: name
  location: location
  tags: tags
  properties: {
    description: 'Alert fires when the Container App running status is not Running (RunningStatus < 1).'
    severity: severity
    enabled: enabled
    scopes: [
      containerAppResourceId
    ]
    evaluationFrequency: 'PT1M'
    windowSize: 'PT5M'
    criteria: {
      'odata.type': 'Microsoft.Azure.Monitor.SingleResourceMultipleMetricCriteria'
      allOf: [
        {
          name: 'RunningStatusNotRunning'
          criterionType: 'StaticThresholdCriterion'
          metricName: 'RunningStatus'
          metricNamespace: 'microsoft.app/containerapps'
          operator: 'LessThan'
          threshold: 1
          timeAggregation: 'Average'
        }
      ]
    }
    autoMitigate: true
  }
}

output alertId string = runningStatusAlert.id
output alertName string = runningStatusAlert.name
