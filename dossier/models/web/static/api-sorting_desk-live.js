/** api-sorting_desk-live.js --- Sorting Desk's live API
 *
 * Copyright (C) 2014 Diffeo
 *
 * Comments:
 *
 *
 */

var QUERY_ID = null;

var _Api = function(window, $, DossierJS) {
    // This initialization is highly suspect.
    var api = new DossierJS.API(DOSSIER_STACK_API_URL),
        qitems = new DossierJS.SortingQueueItems(
            api, 'index_scan', '', 'unknown');

    var getRandomItem = function() {
        var mkitem = function(content_id, fc) {
            return {
                content_id: content_id,
                fc: fc,
                node_id: content_id,
                name: fc.value('NAME') || '',
                text: fc.value('sentences')
                    || (fc.value('NAME') + ' (profile)'),
                url: fc.value('abs_url')
            };
        };
        if (QUERY_ID === null) {
            return api.fcRandomGet().then(function(cobj) {
                return mkitem(cobj[0], cobj[1]);
            });
        } else {
            return api.fcGet(QUERY_ID).then(function(fc) {
                return mkitem(QUERY_ID, fc);
            });
        }
    };

    var setQueryContentId = function (id) {
        if(!id)
            throw "Invalid engine content id";

        qitems.query_content_id = id;
    };

    var setSearchEngine = function(name) {
        qitems.engine_name = name;
    };

    var getSearchEngine = function(name) {
        return qitems.engine_name;
    };

    var itemDroppedInBin = function (item, bin) {
        return api.addLabel(bin.id, item.content.content_id, qitems.annotator, 1);
    };

    var mergeBins = function (ibin, jbin) {
        return api.addLabel(ibin.id, jbin.id, qitems.annotator, 1);
    };

    return $.extend({}, qitems.callbacks(), {
        getRandomItem: getRandomItem,
        setQueryContentId: setQueryContentId,
        itemDroppedInBin: itemDroppedInBin,
        mergeBins: mergeBins,
        setSearchEngine: setSearchEngine,
        getSearchEngine: getSearchEngine,
    });
};

if(typeof define === "function" && define.amd) {
    define("API-SortingDesk", ["jquery", "DossierJS"], function($, DossierJS) {
        return _Api(window, $, DossierJS);
    });
} else {
    var Api = _Api(window, $, DossierJS);
}
