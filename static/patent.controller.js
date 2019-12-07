(function () {
    'use strict';
    var myApp = angular.module('PatentAnalysisApp');
    myApp.controller('PatentController', function ($scope, fileUploadService) {

        $scope.doAnalysis = function(){
            var file = $scope.myFile;
            var uploadUrl = 'http://localhost:5000/analyze_file'   
        };

        $scope.generateStats = function () {
            var file = $scope.myFile;
            var uploadUrl = 'http://localhost:5000/generate_stats',
            promise = fileUploadService.uploadFileToUrl(file, uploadUrl);

            promise.then(function (response) {
                $scope.statsFinished = true;
                $scope.applicants = [];
                response['Applicants'].forEach(function(item){
                    $scope.applicants.push({"name":item[1],"count":item[0]})
                });
                $scope.inventors = [];
                response['Inventors'].forEach(function(item){
                    $scope.inventors.push({"name":item[1],"count":item[0]})
                });
                $scope.jurisdictions = [];
                response['Jurisdiction'].forEach(function(item){
                    $scope.jurisdictions.push({"name":item[1],"count":item[0]})
                });
                $scope.kinds = [];
                response['Kind'].forEach(function(item){
                    $scope.kinds.push({"name":item[1],"count":item[0]})
                });
                $scope.familySizes = [];
                response['Simple Family Size'].forEach(function(item){
                    $scope.familySizes.push({"size":item[1],"count":item[0]})
                });
                $scope.types = [];
                response['Type'].forEach(function(item){
                    $scope.types.push({"name":item[1],"count":item[0]})
                });
            }, function () {
                $scope.serverResponse = 'An error has occurred';
            })
        };
    });

})();