(function () {
    'use strict';
    var app = angular.module('PatentAnalysisApp', []);
    app.config(['$interpolateProvider', function($interpolateProvider) {
        $interpolateProvider.startSymbol('{a');
        $interpolateProvider.endSymbol('a}');
      }]);

})();